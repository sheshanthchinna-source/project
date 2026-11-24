import os
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63


def build_model(class_count):
    return tf.keras.Sequential([
        tf.keras.Input(batch_shape=(1, SEQUENCE_LENGTH, NUM_FEATURES)),
        tf.keras.layers.LSTM(64, return_sequences=True, name="lstm"),
        tf.keras.layers.Dropout(0.2, name="dropout"),
        tf.keras.layers.LSTM(64, name="lstm_1"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense"),
        tf.keras.layers.Dense(class_count, activation="softmax", name="dense_1"),
    ])


def layer_variables(weights, layer):
    layer_key = f"layers\\{layer.name}"
    if layer_key in weights:
        group_key = layer_key
        if isinstance(layer, tf.keras.layers.LSTM):
            group_key += "\\cell"
            group = weights[group_key]
        else:
            group = weights[group_key]
    else:
        group = weights["layers"][layer.name]
        if isinstance(layer, tf.keras.layers.LSTM):
            group = group["cell"]

    variables = group["vars"]
    return [variables[str(index)][...] for index in range(len(layer.get_weights()))]


def convert_model(name, class_count):
    model_dir = BASE_DIR / "model"
    source = model_dir / f"{name}.keras"
    target = model_dir / f"{name}.tflite"
    model = build_model(class_count)

    with zipfile.ZipFile(source) as archive:
        weights_data = archive.read("model.weights.h5")

    descriptor, temporary_path = tempfile.mkstemp(suffix=".weights.h5")
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(weights_data)
        with h5py.File(temporary_path, "r") as weights:
            for layer in model.layers:
                if layer.weights:
                    layer.set_weights(layer_variables(weights, layer))
    finally:
        os.unlink(temporary_path)

    converted = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    interpreter = tf.lite.Interpreter(model_content=converted)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details["shape"].tolist() != [1, SEQUENCE_LENGTH, NUM_FEATURES]:
        raise ValueError(f"{name} has unexpected input shape: {input_details['shape']}")

    sample = np.random.default_rng(42).normal(
        size=(1, SEQUENCE_LENGTH, NUM_FEATURES)
    ).astype(np.float32)
    interpreter.set_tensor(input_details["index"], sample)
    interpreter.invoke()
    actual = interpreter.get_tensor(output_details["index"])
    expected = model(sample, training=False).numpy()
    max_error = float(np.max(np.abs(actual - expected)))
    if max_error > 1e-4:
        raise ValueError(f"{name} conversion differs from Keras by {max_error}")

    target.write_bytes(converted)
    print(f"Converted {source.name} -> {target.name}; max error: {max_error:.2g}")


if __name__ == "__main__":
    convert_model("sign_model", 10)
    convert_model("alphabet_model", 26)

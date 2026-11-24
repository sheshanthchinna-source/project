import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from convert_models import convert_model

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63

def load_dataset(labels, folder_name):
    X, y = [], []
    sample_counts = [0] * len(labels)

    for label, name in enumerate(labels):
        folder = os.path.join(folder_name, name)

        if not os.path.isdir(folder):
            print(f"WARNING: folder not found, skipping: {folder}")
            continue

        count = 0
        for file in os.listdir(folder):
            if not file.endswith(".npy"):
                continue

            data = np.load(os.path.join(folder, file))

            if data.ndim == 2 and data.shape[1] == NUM_FEATURES:
                if data.shape[0] > SEQUENCE_LENGTH:
                    data = data[:SEQUENCE_LENGTH]
                elif data.shape[0] < SEQUENCE_LENGTH:
                    padding = np.zeros(
                        (SEQUENCE_LENGTH - data.shape[0], NUM_FEATURES)
                    )
                    data = np.vstack([data, padding])

                X.append(data)
                y.append(label)
                count += 1
        sample_counts[label] = count

        print(f"{name}: {count} samples")

    if not X:
        raise SystemExit(f"No training data found in {folder_name}")

    insufficient_labels = [
        name for name, count in zip(labels, sample_counts) if count < 2
    ]
    if insufficient_labels:
        raise ValueError(
            "Each class needs at least 2 valid samples; missing or insufficient: "
            + ", ".join(insufficient_labels)
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def train_model(labels, dataset_path, output_path):
    print("\n==============================")
    print("Training:", output_path)
    print("==============================")

    X, y_raw = load_dataset(labels, dataset_path)
    y = to_categorical(y_raw, num_classes=len(labels))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y_raw,
        random_state=42
    )

    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(len(labels), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs("model", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            output_path,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    model.save(output_path)
    print("Saved:", output_path)


# Dataset folder names; app.py maps these class indices to display phrases.
WORDS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "GOOD",
    "LOVE",
    "HELP",
    "QUESTION",
    "WANT TO TALK",
    "ROCK ON",
]
train_model(WORDS, "dataset", "model/sign_model.keras")
convert_model("sign_model", len(WORDS))

# Alphabet model: collect A-Z first, then run this script.
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
missing_alphabets = [
    letter
    for letter in ALPHABETS
    if not os.path.isdir(os.path.join("dataset_alphabet", letter))
    or len([
        file for file in os.listdir(os.path.join("dataset_alphabet", letter))
        if file.endswith(".npy")
    ]) < 2
]
if missing_alphabets:
    print(
        "Skipping alphabet training; collect at least 2 samples for: "
        + ", ".join(missing_alphabets)
    )
else:
    train_model(ALPHABETS, "dataset_alphabet", "model/alphabet_model.keras")
    convert_model("alphabet_model", len(ALPHABETS))

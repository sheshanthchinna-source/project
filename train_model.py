import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


WORDS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP"
]

SEQUENCE_LENGTH = 30

X = []
y = []


for label, word in enumerate(WORDS):

    folder = os.path.join(
        "dataset",
        word
    )

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            data = np.load(
                os.path.join(folder, file)
            )

            if data.shape == (30, 63):

                X.append(data)
                y.append(label)


X = np.array(X)

y = to_categorical(
    y,
    num_classes=len(WORDS)
)


print("Training data:", X.shape)


model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(30, 63)
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(64)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(64, activation="relu")
)

model.add(
    Dense(
        len(WORDS),
        activation="softmax"
    )
)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    X,
    y,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)


os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    "model/sign_model.keras"
)

print("Model saved successfully.")
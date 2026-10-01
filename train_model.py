import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split


WORDS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "GOOD",
    "LOVE"
]

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63  # 21 hand landmarks x 3 coordinates

X = []
y = []


def pad_or_truncate(data, target_len):
    """Pad short sequences with zeros, truncate long ones."""
    if data.shape[0] > target_len:
        return data[:target_len]
    if data.shape[0] < target_len:
        padding = np.zeros((target_len - data.shape[0], data.shape[1]))
        return np.vstack([data, padding])
    return data


# ---------------- Load dataset ----------------

for label, word in enumerate(WORDS):

    folder = os.path.join("dataset", word)

    if not os.path.isdir(folder):
        print(f"WARNING: folder not found, skipping: {folder}")
        continue

    count = 0

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            data = np.load(os.path.join(folder, file))

            # Accept any length, fix to (30, 63) instead of dropping samples
            if data.ndim == 2 and data.shape[1] == NUM_FEATURES:
                data = pad_or_truncate(data, SEQUENCE_LENGTH)
                X.append(data)
                y.append(label)
                count += 1

    print(f"{word}: {count} samples")


X = np.array(X)
y = to_categorical(y, num_classes=len(WORDS))

print("Training data:", X.shape)

if len(X) == 0:
    raise SystemExit("No training data found. Check your dataset folder.")


# ---------------- Proper stratified train/val split ----------------
# (validation_split in model.fit takes the LAST 20% before shuffling,
#  which can leave whole classes out of validation)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=np.argmax(y, axis=1),
    random_state=42
)

print("Train:", X_train.shape, " Validation:", X_val.shape)


# ---------------- Model ----------------

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)
    )
)
model.add(Dropout(0.2))

model.add(LSTM(64))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))
model.add(Dense(len(WORDS), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------------- Callbacks ----------------

os.makedirs("model", exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "model/sign_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)


# ---------------- Final evaluation ----------------

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal validation accuracy: {val_acc * 100:.2f}%")

model.save("model/sign_model_final.keras")
print("Model saved successfully.")

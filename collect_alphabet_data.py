import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Collect alphabet sequences A-Z for the alphabet LSTM.
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SEQUENCE_LENGTH = 30
SAMPLES_PER_LETTER = 60
DATA_PATH = "dataset_alphabet"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def extract_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for point in hand.landmark:
            landmarks.extend([point.x, point.y, point.z])
        return landmarks

    # Keep the sequence shape valid when the hand disappears.
    return np.zeros(63, dtype=np.float32)

for letter in ALPHABETS:
    letter_path = os.path.join(DATA_PATH, letter)
    os.makedirs(letter_path, exist_ok=True)

    print(f"\nAlphabet: {letter}")
    print("Press SPACE to start recording this letter.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Show: {letter} | SPACE = start | Q = quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (255, 255, 255), 2)
        cv2.imshow("Alphabet Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    for sample in range(SAMPLES_PER_LETTER):
        frames = []
        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            landmarks = extract_landmarks(frame)
            frames.append(landmarks)

            cv2.putText(frame, f"{letter}: {sample + 1}/{SAMPLES_PER_LETTER}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        (255, 255, 255), 2)
            cv2.imshow("Alphabet Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        np.save(
            os.path.join(letter_path, f"{sample}.npy"),
            np.asarray(frames, dtype=np.float32)
        )
        time.sleep(0.08)

    print(f"Completed {letter}")

cap.release()
cv2.destroyAllWindows()
print("Alphabet data collection completed.")

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Words to train
WORDS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP"
]

SEQUENCE_LENGTH = 30
DATA_PATH = "dataset"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)


def extract_landmarks(frame):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for point in hand.landmark:
            landmarks.append(point.x)
            landmarks.append(point.y)
            landmarks.append(point.z)

        return landmarks

    return np.zeros(63)


for word in WORDS:

    word_path = os.path.join(
        DATA_PATH,
        word
    )

    os.makedirs(
        word_path,
        exist_ok=True
    )

    print()
    print("Word:", word)
    print("Press SPACE to start recording")

    while True:

        ret, frame = cap.read()

        cv2.putText(
            frame,
            f"Show: {word}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Data Collection",
            frame
        )

        key = cv2.waitKey(1)

        if key == 32:
            break

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()


    # Record 40 sequences
    for sequence in range(40):

        frames = []

        print(
            f"Recording {word}: "
            f"{sequence + 1}/40"
        )

        start_time = time.time()

        while len(frames) < SEQUENCE_LENGTH:

            ret, frame = cap.read()

            landmarks = extract_landmarks(frame)

            frames.append(landmarks)

            cv2.putText(
                frame,
                word,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Data Collection",
                frame
            )

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        file_name = os.path.join(
            word_path,
            f"{sequence}.npy"
        )

        np.save(
            file_name,
            np.array(frames)
        )

        time.sleep(0.2)


cap.release()

cv2.destroyAllWindows()

print("Data collection completed.")
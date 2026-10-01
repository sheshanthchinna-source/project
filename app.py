from flask import Flask, render_template, request, jsonify

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import threading
from collections import deque, Counter


app = Flask(__name__)


MODEL_PATH = "model/sign_model.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

WORDS = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "GOOD",
    "LOVE"
]

SEQUENCE_LENGTH = 30

CONFIDENCE_THRESHOLD = 0.75   # 6 classes: 0.45 is barely above a guess

STABLE_FRAMES = 5            # word must win 5 predictions in a row-window

MAX_NO_HAND_FRAMES = 3
PREDICT_EVERY = 3            # run the model every 3rd frame (faster)
lock = threading.Lock()
frame_counter = 0

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
sequence = []

prediction_history = []

last_word = "Unknown"

no_hand_count = 0
def extract_landmarks(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    # No hand detected
    if not results.multi_hand_landmarks:

        return None

    hand = results.multi_hand_landmarks[0]

    landmarks = []

    for point in hand.landmark:

        landmarks.append(point.x)
        landmarks.append(point.y)
        landmarks.append(point.z)

    return landmarks

def predict_word():
    global last_word

    if len(sequence) != SEQUENCE_LENGTH:
        return last_word, 0.0

    input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
    # model(...) is much faster than model.predict(...) for a single sample
    prediction = model(input_data, training=False).numpy()[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index])
    word = WORDS[index]

    if confidence < CONFIDENCE_THRESHOLD:
        prediction_history.append(None)
    else:
        prediction_history.append(word)
    if len(prediction_history) > STABLE_FRAMES:
        prediction_history.pop(0)

    # Only accept a word when it is the SAME for all recent predictions.
    # Otherwise keep showing the last confirmed word (never a random guess).
    if (len(prediction_history) == STABLE_FRAMES
            and len(set(prediction_history)) == 1
            and prediction_history[0] is not None):
        last_word = word
        return word, confidence

    return last_word, confidence

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    global sequence
    global no_hand_count
    global prediction_history
    global last_word
    global frame_counter

    try:

        data = request.get_json()

        if not data or "image" not in data:

            return jsonify({
                "error": "No image received"
            }), 400

        image_data = data["image"]

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        image_bytes = base64.b64decode(
            image_data
        )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({
                "error": "Invalid image"
            }), 400

        frame = cv2.resize(frame, (320, 240))
        with lock:
            return handle_frame(frame)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


def handle_frame(frame):
    global no_hand_count, last_word, frame_counter
    if True:
        landmarks = extract_landmarks(
            frame
        )


        if landmarks is None:

            no_hand_count += 1
            prediction_history.clear()

            # Don't allow old movement to remain forever
            if no_hand_count >= MAX_NO_HAND_FRAMES:

                sequence.clear()
                prediction_history.clear()

                last_word = "Unknown"


            return jsonify({

                "word": "No hand detected",

                "confidence": 0,

                "hand_detected": False,

                "frames": len(sequence)

            })

        no_hand_count = 0


        # Add valid landmarks only
        sequence.append(
            landmarks
        )


        # Keep only latest 30 frames
        if len(sequence) > SEQUENCE_LENGTH:

            sequence.pop(0)


        if len(sequence) < SEQUENCE_LENGTH:

            return jsonify({

                "word": "Collecting movement...",

                "confidence": 0,

                "hand_detected": True,

                "frames": len(sequence)

            })


        frame_counter += 1
        if frame_counter % PREDICT_EVERY != 0:
            return jsonify({"word": last_word, "confidence": 0,
                            "hand_detected": True, "frames": len(sequence),
                            "skipped": True})

        word, confidence = predict_word()


        return jsonify({

            "word": word,

            "confidence": round(
                confidence,
                3
            ),

            "hand_detected": True,

            "frames": len(sequence)

        })




if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False,  # avoid loading the model twice
        threaded=False,
        host="127.0.0.1",
        port=5000
    )
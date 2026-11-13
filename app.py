from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import threading
from collections import deque

app = Flask(__name__)

# ---------------- MODELS ----------------
WORD_MODEL_PATH = "model/sign_model.keras"
ALPHABET_MODEL_PATH = "model/alphabet_model.keras"

word_model = tf.keras.models.load_model(WORD_MODEL_PATH, compile=False)

try:
    alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH, compile=False)
    ALPHABET_MODEL_READY = True
except Exception as e:
    print("Alphabet model not loaded:", e)
    alphabet_model = None
    ALPHABET_MODEL_READY = False

WORDS = ["HELLO How Are You?", "THANK YOU FOR HELP!", "YES,I'M DONE", "NO,I'M NOT INTERSTED", "LOOKING GOOD", "LOVE YOU","PLEASE! HELP TO ME.."]
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAMES = 5
MAX_NO_HAND_FRAMES = 3
PREDICT_EVERY = 3

lock = threading.Lock()
frame_counter = 0
sequence = []
no_hand_count = 0

word_history = deque(maxlen=STABLE_FRAMES)
alphabet_history = deque(maxlen=STABLE_FRAMES)
word_score_history = deque(maxlen=STABLE_FRAMES)
alphabet_score_history = deque(maxlen=STABLE_FRAMES)
last_word = "Unknown"
last_alphabet = "Unknown"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    landmarks = []

    for point in hand.landmark:
        landmarks.extend([point.x, point.y, point.z])

    return landmarks


def stable_prediction(model, labels, history, score_history, previous):
    """
    Smooth model probabilities across recent predictions, then require a stable
    label before changing the accepted result. Returns the accepted label and
    smoothed confidence for that label.
    """
    input_data = np.expand_dims(
        np.array(sequence, dtype=np.float32), axis=0
    )

    scores = np.asarray(
        model(input_data, training=False).numpy()[0],
        dtype=np.float32
    )

    # Keep a rolling average of class probabilities to reduce frame-to-frame noise.
    score_history.append(scores)
    smoothed_scores = np.mean(np.stack(score_history, axis=0), axis=0)

    index = int(np.argmax(smoothed_scores))
    confidence = float(smoothed_scores[index])
    label = labels[index]

    if confidence >= CONFIDENCE_THRESHOLD:
        history.append(label)
    else:
        history.append(None)

    # Switch to a new label only when every recent valid result agrees.
    if (
        len(history) == STABLE_FRAMES
        and all(item is not None for item in history)
        and len(set(history)) == 1
    ):
        return label, confidence

    # Preserve the last stable label while reporting smoothed confidence.
    return previous, confidence


@app.route("/")
def home():
    return render_template("index.html")
@app.route('/asl-alphabet')
def asl_alphabet():
    return render_template('asl-alphabet.html')

@app.route("/predict", methods=["POST"])
def predict():
    global sequence, no_hand_count, frame_counter
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image received"}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        frame = cv2.resize(frame, (320, 240))

        with lock:
            return handle_frame(frame)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


def handle_frame(frame):
    global no_hand_count, last_word, last_alphabet, frame_counter

    landmarks = extract_landmarks(frame)

    if landmarks is None:
        no_hand_count += 1
        word_history.clear()
        alphabet_history.clear()
        word_score_history.clear()
        alphabet_score_history.clear()

        if no_hand_count >= MAX_NO_HAND_FRAMES:
            sequence.clear()
            word_history.clear()
            alphabet_history.clear()
            word_score_history.clear()
            alphabet_score_history.clear()
            last_word = "Unknown"
            last_alphabet = "Unknown"

        return jsonify({
            "word": "No hand detected",
            "alphabet": "No hand detected",
            "confidence": 0,
            "alphabet_confidence": 0,
            "hand_detected": False,
            "frames": len(sequence),
            "alphabet_model_ready": ALPHABET_MODEL_READY
        })

    no_hand_count = 0
    sequence.append(landmarks)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) < SEQUENCE_LENGTH:
        return jsonify({
            "word": "Collecting movement...",
            "alphabet": "Collecting...",
            "confidence": 0,
            "alphabet_confidence": 0,
            "hand_detected": True,
            "frames": len(sequence),
            "alphabet_model_ready": ALPHABET_MODEL_READY
        })

    frame_counter += 1
    if frame_counter % PREDICT_EVERY != 0:
        return jsonify({
            "word": last_word,
            "alphabet": last_alphabet,
            "confidence": 0,
            "alphabet_confidence": 0,
            "hand_detected": True,
            "frames": len(sequence),
            "skipped": True,
            "alphabet_model_ready": ALPHABET_MODEL_READY
        })

    last_word, word_conf = stable_prediction(
        word_model, WORDS, word_history, word_score_history, last_word
    )

    alphabet_conf = 0.0
    if ALPHABET_MODEL_READY:
        last_alphabet, alphabet_conf = stable_prediction(
            alphabet_model, ALPHABETS, alphabet_history, alphabet_score_history, last_alphabet
        )
    else:
        last_alphabet = "Model not trained"

    return jsonify({
        "word": last_word,
        "alphabet": last_alphabet,
        "confidence": round(word_conf, 3),
        "alphabet_confidence": round(alphabet_conf, 3),
        "hand_detected": True,
        "frames": len(sequence),
        "alphabet_model_ready": ALPHABET_MODEL_READY
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

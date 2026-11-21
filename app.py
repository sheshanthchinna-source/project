import os
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

BASE_DIR = Path(__file__).resolve().parent
WORD_MODEL_PATH = BASE_DIR / "model" / "sign_model.tflite"
ALPHABET_MODEL_PATH = BASE_DIR / "model" / "alphabet_model.tflite"

WORDS = [
    "HELLO How Are You?",
    "THANK YOU FOR HELP!",
    "YES,I'M DONE",
    "NO,I'M NOT INTERSTED",
    "LOOKING GOOD",
    "LOVE YOU",
    "PLEASE! HELP TO ME..",
]
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAMES = 5

word_model = None
alphabet_model = None
ALPHABET_MODEL_READY = False
_model_lock = threading.Lock()


class TFLiteModel:
    def __init__(self, path):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(path), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)


def default_state():
    return {
        "word_history": [],
        "alphabet_history": [],
        "word_score_history": [],
        "alphabet_score_history": [],
        "last_word": "Unknown",
        "last_alphabet": "Unknown",
    }


def parse_client_state(value):
    state = default_state()
    if value is None:
        return state
    if not isinstance(value, dict):
        raise ValueError("state must be an object")

    labels = {
        "word_history": WORDS,
        "alphabet_history": ALPHABETS,
    }
    for key, allowed_labels in labels.items():
        history = value.get(key, [])
        if (
            not isinstance(history, list)
            or len(history) > STABLE_FRAMES
            or any(item is not None and item not in allowed_labels for item in history)
        ):
            raise ValueError(f"state.{key} must contain at most {STABLE_FRAMES} valid labels")
        state[key] = history

    class_counts = {
        "word_score_history": len(WORDS),
        "alphabet_score_history": len(ALPHABETS),
    }
    for key, class_count in class_counts.items():
        history = value.get(key, [])
        if not isinstance(history, list) or len(history) > STABLE_FRAMES:
            raise ValueError(f"state.{key} must contain at most {STABLE_FRAMES} score arrays")
        for scores in history:
            if (
                not isinstance(scores, list)
                or len(scores) != class_count
                or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not np.isfinite(score)
                    or score < 0
                    or score > 1
                    for score in scores
                )
            ):
                raise ValueError(f"state.{key} contains an invalid score array")
        state[key] = history

    for key, allowed_labels in (
        ("last_word", WORDS + ["Unknown"]),
        ("last_alphabet", ALPHABETS + ["Unknown", "Model not trained"]),
    ):
        previous = value.get(key, "Unknown")
        if not isinstance(previous, str) or previous not in allowed_labels:
            raise ValueError(f"state.{key} is invalid")
        state[key] = previous

    return state


def validate_sequence(value):
    if not isinstance(value, list) or len(value) != SEQUENCE_LENGTH:
        raise ValueError(f"sequence must contain exactly {SEQUENCE_LENGTH} frames")

    for frame in value:
        if not isinstance(frame, list) or len(frame) != NUM_FEATURES:
            raise ValueError(f"each sequence frame must contain exactly {NUM_FEATURES} values")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not np.isfinite(coordinate)
            for coordinate in frame
        ):
            raise ValueError("sequence coordinates must be finite numbers")

    return np.asarray(value, dtype=np.float32)


def load_models():
    global word_model, alphabet_model, ALPHABET_MODEL_READY
    if word_model is not None:
        return

    with _model_lock:
        if word_model is not None:
            return

        word_model = TFLiteModel(WORD_MODEL_PATH)
        try:
            alphabet_model = TFLiteModel(ALPHABET_MODEL_PATH)
            ALPHABET_MODEL_READY = True
        except Exception as exc:
            print("Alphabet model not loaded:", exc)
            alphabet_model = None
            ALPHABET_MODEL_READY = False


def stable_prediction(model, labels, sequence, history, score_history, previous):
    scores = np.asarray(
        model.predict(np.expand_dims(sequence, axis=0))[0],
        dtype=np.float32,
    )
    score_history.append(scores.tolist())
    score_history = score_history[-STABLE_FRAMES:]
    smoothed_scores = np.mean(np.asarray(score_history, dtype=np.float32), axis=0)

    index = int(np.argmax(smoothed_scores))
    confidence = float(smoothed_scores[index])
    label = labels[index]
    history.append(label if confidence >= CONFIDENCE_THRESHOLD else None)
    history = history[-STABLE_FRAMES:]

    if (
        len(history) == STABLE_FRAMES
        and all(item is not None for item in history)
        and len(set(history)) == 1
    ):
        return label, confidence, history, score_history
    return previous, confidence, history, score_history


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/asl-alphabet")
def asl_alphabet():
    return render_template("asl-alphabet.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    if "sequence" not in data:
        return jsonify({"error": "No sequence received"}), 400

    try:
        sequence = validate_sequence(data["sequence"])
        state = parse_client_state(data.get("state"))
    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        load_models()
        with _model_lock:
            (
                state["last_word"],
                word_confidence,
                state["word_history"],
                state["word_score_history"],
            ) = stable_prediction(
                word_model,
                WORDS,
                sequence,
                state["word_history"],
                state["word_score_history"],
                state["last_word"],
            )

            alphabet_confidence = 0.0
            if ALPHABET_MODEL_READY:
                (
                    state["last_alphabet"],
                    alphabet_confidence,
                    state["alphabet_history"],
                    state["alphabet_score_history"],
                ) = stable_prediction(
                    alphabet_model,
                    ALPHABETS,
                    sequence,
                    state["alphabet_history"],
                    state["alphabet_score_history"],
                    state["last_alphabet"],
                )
            else:
                state["last_alphabet"] = "Model not trained"

        return jsonify({
            "word": state["last_word"],
            "alphabet": state["last_alphabet"],
            "confidence": round(word_confidence, 3),
            "alphabet_confidence": round(alphabet_confidence, 3),
            "hand_detected": True,
            "frames": SEQUENCE_LENGTH,
            "alphabet_model_ready": ALPHABET_MODEL_READY,
            "state": state,
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        use_reloader=False,
        threaded=True,
    )

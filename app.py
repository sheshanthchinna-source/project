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
CONFIDENCE_THRESHOLD = 0.50
STABLE_FRAMES = 3

word_model = None
alphabet_model = None
ALPHABET_MODEL_READY = False
_model_lock = threading.Lock()


class TFLiteModel:
    def __init__(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter

        self.interpreter = Interpreter(model_path=str(path), num_threads=2)
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        self.input_index = inp["index"]
        self.output_index = out["index"]
        self.input_shape = tuple(inp["shape"])
        self.input_dtype = inp["dtype"]
        self.output_shape = tuple(out["shape"])
        self.input_quantization = inp.get("quantization", (0.0, 0))
        print(f"Loaded {path.name}: input={self.input_shape}, output={self.output_shape}")

    def predict(self, input_data):
        data = np.asarray(input_data, dtype=np.float32)
        if tuple(data.shape) != self.input_shape:
            try:
                data = data.reshape(self.input_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Model expects {self.input_shape}; received {data.shape}"
                ) from exc
        if np.issubdtype(self.input_dtype, np.integer):
            scale, zero_point = self.input_quantization
            if scale:
                data = np.round(data / scale + zero_point)
            limits = np.iinfo(self.input_dtype)
            data = np.clip(data, limits.min, limits.max).astype(self.input_dtype)
        else:
            data = data.astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_index, data)
        self.interpreter.invoke()
        return np.asarray(self.interpreter.get_tensor(self.output_index), dtype=np.float32)


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
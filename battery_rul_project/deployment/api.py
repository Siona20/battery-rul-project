"""
Battery Health Prediction API — FINAL FIXED VERSION
"""

import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "outputs", "models"))
SCALER_DIR = os.path.join(BASE_DIR, "scalers")

# ✅ USE THESE FILES (you confirmed they exist)
MODEL_PATH_SOH = os.path.join(MODEL_DIR, "soh_best.keras")
MODEL_PATH_RUL = os.path.join(MODEL_DIR, "rul_best.keras")

print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("SCALER_DIR:", SCALER_DIR)
print("TF VERSION:", tf.__version__)

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
PHYSICAL_FEATURES = [
    "avg_voltage",
    "min_voltage",
    "avg_current",
    "duration",
    "voltage_drop",
    "temp_change",
    "cycle",
    "cumul_voltage_drop",
    "cycle_rank",
]

rolling_feats = []
for f in PHYSICAL_FEATURES:
    rolling_feats.extend([f"{f}_mean", f"{f}_std"])

ALL_FEATURES = PHYSICAL_FEATURES + rolling_feats

# ─────────────────────────────────────────────
# LOAD MODELS + SCALERS
# ─────────────────────────────────────────────
def load_assets():
    try:
        print("\n===== LOADING ASSETS =====")

        # CHECK DIRECTORIES
        if not os.path.isdir(MODEL_DIR):
            raise Exception(f"Model directory not found: {MODEL_DIR}")

        if not os.path.isdir(SCALER_DIR):
            raise Exception(f"Scaler directory not found: {SCALER_DIR}")

        print("Model files:", os.listdir(MODEL_DIR))
        print("Scaler files:", os.listdir(SCALER_DIR))

        # LOAD MODELS (TF SAFE MODE FIX)
        try:
            soh_model = tf.keras.models.load_model(
                MODEL_PATH_SOH, compile=False, safe_mode=False
            )
            rul_model = tf.keras.models.load_model(
                MODEL_PATH_RUL, compile=False, safe_mode=False
            )
        except TypeError:
            # fallback for older TF
            soh_model = tf.keras.models.load_model(MODEL_PATH_SOH, compile=False)
            rul_model = tf.keras.models.load_model(MODEL_PATH_RUL, compile=False)

        print("✅ Models loaded")

        # LOAD SCALERS
        scalers = {
            "soh_feat": joblib.load(os.path.join(SCALER_DIR, "soh_feat_scaler.pkl")),
            "rul_feat": joblib.load(os.path.join(SCALER_DIR, "rul_feat_scaler.pkl")),
            "rul_tgt": joblib.load(os.path.join(SCALER_DIR, "rul_tgt_scaler.pkl")),
        }

        print("✅ Scalers loaded")
        print("===== SUCCESS =====\n")

        return soh_model, rul_model, scalers

    except Exception as e:
        print("\n❌ MODEL LOADING FAILED:")
        traceback.print_exc()
        return None, None, None


# LOAD ON START
soh_model, rul_model, scalers = load_assets()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_feature_row(data):
    cycle = float(data.get("cycle", 1))
    voltage_drop = float(data["voltage_drop"])

    raw = {
        "avg_voltage": float(data["avg_voltage"]),
        "min_voltage": float(data["min_voltage"]),
        "avg_current": float(data["avg_current"]),
        "duration": float(data["duration"]),
        "voltage_drop": voltage_drop,
        "temp_change": float(data["temp_change"]),
        "cycle": cycle,
        "cumul_voltage_drop": cycle * voltage_drop,
        "cycle_rank": float(data.get("cycle_rank", 0.5)),
    }

    row = {}
    for f in PHYSICAL_FEATURES:
        row[f] = raw[f]
        row[f"{f}_mean"] = raw[f]
        row[f"{f}_std"] = 0.0

    return pd.DataFrame([row])[ALL_FEATURES]


def make_sequence(row, seq_len):
    return np.repeat(row, seq_len, axis=0).reshape(1, seq_len, -1)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "models_loaded": soh_model is not None and rul_model is not None,
        "scalers_loaded": scalers is not None
    })


@app.route("/debug")
def debug():
    return jsonify({
        "model_dir": MODEL_DIR,
        "model_files": os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else [],
        "scaler_dir": SCALER_DIR,
        "scaler_files": os.listdir(SCALER_DIR) if os.path.exists(SCALER_DIR) else [],
        "soh_model_exists": os.path.exists(MODEL_PATH_SOH),
        "rul_model_exists": os.path.exists(MODEL_PATH_RUL),
        "models_loaded": soh_model is not None,
        "scalers_loaded": scalers is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if soh_model is None or rul_model is None:
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.get_json()

        df = build_feature_row(data)

        # SOH
        soh_scaled = scalers["soh_feat"].transform(df)
        seq_soh = make_sequence(soh_scaled, SEQ_LEN)
        soh = soh_model.predict(seq_soh, verbose=0)[0][0] * 100

        # RUL
        rul_scaled = scalers["rul_feat"].transform(df)
        seq_rul = make_sequence(rul_scaled, SEQ_LEN)
        rul_pred = rul_model.predict(seq_rul, verbose=0)
        rul = scalers["rul_tgt"].inverse_transform(rul_pred)[0][0]

        return jsonify({
            "SOH": round(float(soh), 2),
            "RUL": round(float(rul), 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
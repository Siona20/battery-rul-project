"""
Battery Health Prediction API — FINAL FIXED VERSION
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ CORRECT PATHS
MODEL_PATH_SOH = os.path.join(BASE_DIR, "..", "model", "outputs", "models", "soh_model.keras")
MODEL_PATH_RUL = os.path.join(BASE_DIR, "..", "model", "outputs", "models", "rul_model.keras")

SCALER_DIR = os.path.join(BASE_DIR, "scalers")

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
    print("🔍 DEBUG START")

    print("BASE_DIR:", BASE_DIR)
    print("SOH MODEL PATH:", MODEL_PATH_SOH)
    print("RUL MODEL PATH:", MODEL_PATH_RUL)
    print("SCALER DIR:", SCALER_DIR)

    # ✅ Check existence FIRST
    if not os.path.exists(MODEL_PATH_SOH):
        raise Exception(f"SOH model not found: {MODEL_PATH_SOH}")

    if not os.path.exists(MODEL_PATH_RUL):
        raise Exception(f"RUL model not found: {MODEL_PATH_RUL}")

    if not os.path.exists(SCALER_DIR):
        raise Exception(f"Scaler dir not found: {SCALER_DIR}")

    print("📁 Model files:", os.listdir(os.path.dirname(MODEL_PATH_SOH)))
    print("📁 Scaler files:", os.listdir(SCALER_DIR))

    # ✅ Load models
    soh_model = tf.keras.models.load_model(MODEL_PATH_SOH, compile=False)
    rul_model = tf.keras.models.load_model(MODEL_PATH_RUL, compile=False)

    # ✅ FIXED SCALER NAMES (MATCH YOUR GITHUB)
    scalers = {
        "soh_feat": joblib.load(os.path.join(SCALER_DIR, "soh_feat_scaler.pkl")),
        "rul_feat": joblib.load(os.path.join(SCALER_DIR, "rul_feat_scaler.pkl")),
        "rul_tgt": joblib.load(os.path.join(SCALER_DIR, "rul_tgt_scaler.pkl")),
    }

    print("✅ EVERYTHING LOADED SUCCESSFULLY")
    return soh_model, rul_model, scalers


# 🚨 IMPORTANT: DO NOT SILENCE ERRORS
try:
    soh_model, rul_model, scalers = load_assets()
except Exception as e:
    print("❌ CRITICAL ERROR:", str(e))
    soh_model, rul_model, scalers = None, None, None


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
        "models_loaded": soh_model is not None and rul_model is not None
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
        soh = soh_model.predict(seq_soh)[0][0] * 100

        # RUL
        rul_scaled = scalers["rul_feat"].transform(df)
        seq_rul = make_sequence(rul_scaled, SEQ_LEN)
        rul_pred = rul_model.predict(seq_rul)
        rul = scalers["rul_tgt"].inverse_transform(rul_pred)[0][0]

        return jsonify({
            "SOH": round(float(soh), 2),
            "RUL": round(float(rul), 2)
        })

    except Exception as e:
        print("❌ PREDICT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
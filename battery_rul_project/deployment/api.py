"""
Battery Health Prediction API — v4
====================================
Flask REST API for LSTM-based SoH and RUL prediction.
Compatible with LSTM model v4 (27-feature input: 9 physical + 18 rolling).

ROOT CAUSE FIX (500 error):
  Training code builds rolling_feats by interleaving per feature:
      for feat in PHYSICAL_FEATURES:
          rolling_feats.extend([f"{feat}_mean", f"{feat}_std"])   # interleaved!
  So ALL_FEATURES column order is:
      [raw_0..8, feat0_mean, feat0_std, feat1_mean, feat1_std, ...]
  The previous API incorrectly used [all_means..., all_stds...] which
  caused sklearn's "feature names must match fit order" 500 error.
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
# CONFIG — mirror v4 training constants exactly
# ─────────────────────────────────────────────
SEQ_LEN           = 20    # must match SEQ_LEN in model_v4.py
FAILURE_THRESHOLD = 0.70

# Paths — models/ and scalers/ sit next to api.py inside deployment/
MODEL_PATH_SOH = os.path.join("models", "soh_model.keras")
MODEL_PATH_RUL = os.path.join("models", "rul_model.keras")
SCALER_DIR     = "scalers"

# 9 physical features — exact order from training
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

# ── CRITICAL: interleaved rolling features, matching training exactly ──────
# Training loop in model_v4.py:
#   for feat in PHYSICAL_FEATURES:
#       rolling_feats.extend([f"{feat}_mean", f"{feat}_std"])  # interleaved!
#   ALL_FEATURES = PHYSICAL_FEATURES + rolling_feats
#
# Resulting order: [raw_0..8,
#                   avg_voltage_mean, avg_voltage_std,
#                   min_voltage_mean, min_voltage_std, ...]  — 27 cols total
_rolling_feats = []
for _f in PHYSICAL_FEATURES:
    _rolling_feats.extend([f"{_f}_mean", f"{_f}_std"])

ALL_FEATURES = PHYSICAL_FEATURES + _rolling_feats  # 9 + 18 = 27


# ─────────────────────────────────────────────
# ASSET LOADING
# ─────────────────────────────────────────────
def load_assets():
    """Load Keras models and joblib scalers from disk."""
    try:
        soh_model = tf.keras.models.load_model(MODEL_PATH_SOH)
        rul_model = tf.keras.models.load_model(MODEL_PATH_RUL)
        scalers = {
            "soh_feat": joblib.load(os.path.join(SCALER_DIR, "soh_feat_scaler.pkl")),
            "rul_feat": joblib.load(os.path.join(SCALER_DIR, "rul_feat_scaler.pkl")),
            "rul_tgt":  joblib.load(os.path.join(SCALER_DIR, "rul_tgt_scaler.pkl")),
        }
        print("[API] Models and scalers loaded successfully.")
        return soh_model, rul_model, scalers
    except Exception as e:
        print(f"[API] ERROR loading assets: {e}")
        return None, None, None


soh_model, rul_model, scalers = load_assets()

# Startup diagnostic — verify column order matches scaler at boot time
print(f"[API] ALL_FEATURES ({len(ALL_FEATURES)}): {ALL_FEATURES}")
if scalers and hasattr(scalers["soh_feat"], "feature_names_in_"):
    fitted = list(scalers["soh_feat"].feature_names_in_)
    if fitted == ALL_FEATURES:
        print("[API] Column order matches scaler — OK.")
    else:
        print("[API] COLUMN MISMATCH — scaler expects:")
        print(f"       {fitted}")
        print("[API]    API is sending:")
        print(f"       {ALL_FEATURES}")


# ─────────────────────────────────────────────
# FEATURE CONSTRUCTION
# ─────────────────────────────────────────────
def build_feature_row(data: dict) -> pd.DataFrame:
    """
    Construct a single 27-feature row from user-supplied values.

    Column order: [raw_0..8, feat0_mean, feat0_std, feat1_mean, feat1_std, ...]
    This exactly mirrors how the scalers were fitted during training.

    Derived features (not directly measurable in real-time):
      - cumul_voltage_drop  ≈ cycle x voltage_drop  (proxy for lifetime wear)
      - cycle_rank          = 0.5  (neutral prior; actual rank unknown at inference)

    Rolling statistics (single-sample approximation):
      - mean = raw value  (window of 1 => mean equals value)
      - std  = 0.0        (no history => zero dispersion)
    """
    cycle              = float(data.get("cycle", 1))
    voltage_drop       = float(data["voltage_drop"])
    cumul_voltage_drop = cycle * voltage_drop
    cycle_rank         = float(data.get("cycle_rank", 0.5))

    raw = {
        "avg_voltage":        float(data["avg_voltage"]),
        "min_voltage":        float(data["min_voltage"]),
        "avg_current":        float(data["avg_current"]),
        "duration":           float(data["duration"]),
        "voltage_drop":       voltage_drop,
        "temp_change":        float(data["temp_change"]),
        "cycle":              cycle,
        "cumul_voltage_drop": cumul_voltage_drop,
        "cycle_rank":         cycle_rank,
    }

    # Build all 27 columns interleaved (raw, mean, std per feature)
    # then select via ALL_FEATURES to guarantee correct order.
    row = {}
    for f in PHYSICAL_FEATURES:
        row[f]           = raw[f]
        row[f"{f}_mean"] = raw[f]  # rolling mean approximated as current value
        row[f"{f}_std"]  = 0.0     # no history -> zero std

    return pd.DataFrame([row])[ALL_FEATURES]   # enforces exact column order


def make_sequence(scaled_row: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Tile a single scaled feature row seq_len times -> (1, seq_len, n_feat).
    Standard real-time inference strategy when no historical window exists.
    """
    return np.repeat(scaled_row, seq_len, axis=0).reshape(1, seq_len, -1)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Liveness probe — returns model load status."""
    loaded = soh_model is not None and rul_model is not None
    return jsonify({
        "status":        "ok" if loaded else "degraded",
        "models_loaded": loaded,
        "seq_len":       SEQ_LEN,
        "n_features":    len(ALL_FEATURES),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
      {
        "avg_voltage":  3.7,
        "min_voltage":  3.2,
        "avg_current":  1.5,
        "duration":     3600,
        "voltage_drop": 0.5,
        "temp_change":  10.0,
        "cycle":        150        // optional, defaults to 1
      }

    Response:
      {
        "SOH":    82.35,           // percentage [0-100]
        "RUL":    312.0,           // remaining cycles
        "status": "Healthy"        // Healthy | Degraded | Critical
      }
    """
    if soh_model is None or rul_model is None:
        return jsonify({"error": "Models not loaded. Check server logs."}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    required = ["avg_voltage", "min_voltage", "avg_current",
                "duration", "voltage_drop", "temp_change"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        df_input = build_feature_row(data)

        # SoH prediction
        scaled_soh = scalers["soh_feat"].transform(df_input)
        seq_soh    = make_sequence(scaled_soh, SEQ_LEN)
        soh_raw    = float(np.clip(
            soh_model.predict(seq_soh, verbose=0).flatten()[0], 0.0, 1.0
        ))
        soh_pct = round(soh_raw * 100, 2)

        # RUL prediction
        scaled_rul      = scalers["rul_feat"].transform(df_input)
        seq_rul         = make_sequence(scaled_rul, SEQ_LEN)
        rul_scaled_pred = rul_model.predict(seq_rul, verbose=0)
        rul_cycles      = float(
            scalers["rul_tgt"].inverse_transform(rul_scaled_pred).flatten()[0]
        )
        rul_cycles = round(max(0.0, rul_cycles), 1)

        # Status label
        if soh_pct > 80:
            status = "Healthy"
        elif soh_pct >= 70:
            status = "Degraded"
        else:
            status = "Critical"

        return jsonify({"SOH": soh_pct, "RUL": rul_cycles, "status": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
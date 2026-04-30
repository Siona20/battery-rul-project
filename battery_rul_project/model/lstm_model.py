"""
LSTM-Based Battery SoH and RUL Prediction — v5
===============================================
Improvements over v4:
  - Split changed: 70/15/15 → 70/20/10 (more validation batteries)
  - cycle_norm lifecycle anchor feature added
  - RUL target normalized: remaining_cycles → rul_ratio
  - Sample weighting for RUL (early cycles weighted higher)
  - RUL model simplified: 3→2 LSTM layers, 256→64 units
  - SoH loss: MSE → Huber(delta=0.1)
  - Validation evaluation logging added
  - Prediction CSVs saved (soh_predictions.csv, rul_predictions.csv)

Future improvements (not implemented):
  # TODO: Add Attention mechanism on top of LSTM for interpretability
  # TODO: Multi-task learning — combine SoH + RUL into a single model
  # TODO: Cross-validation using GroupKFold for robust evaluation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import json
import math

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
SEED              = 42    # reproducibility — unchanged from v3

SEQ_LEN           = 20   # increased from 10: 20 past cycles gives the model
                          # enough context to distinguish early vs late degradation.
                          # With 10, batteries like battery01 (1484 cycles) had
                          # almost no positional signal per window.

BATCH             = 32   # kept from v3: with ~10,000 train sequences and 16
                          # batteries, batch=32 gives ~312 gradient steps/epoch —
                          # already stable. Increasing to 64 halves steps/epoch
                          # with no benefit on this dataset size.

EPOCHS            = 50   # kept from v3: EarlyStopping controls actual stopping.
                          # v3 stopped at epoch 45 (SoH) and 32 (RUL), so 50
                          # is sufficient ceiling. 100 is unnecessary overhead.

PATIENCE          = 20   # increased from 15: v3 val_loss oscillated heavily
                          # (visible in training plots — spikes at epochs 3, 10).
                          # 15 risked stopping during a temporary spike before
                          # the model recovered. 20 gives it more room.

FAILURE_THRESHOLD = 0.70  # domain constant — SoH < 70% = End of Life.
                           # unchanged from v3, not a tunable hyperparameter.

DATA_PATH  = "../processed_data/final_dataset_with_soh.csv"
OUT_DIR    = "../model/outputs"
MODEL_DIR  = "../model/outputs/models"
SCALER_DIR = "../model/outputs/scalers"

for d in [OUT_DIR, MODEL_DIR, SCALER_DIR]:
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  LSTM Battery SoH & RUL Prediction Pipeline v5")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["remaining_cycles"] = (1 - df["cycle_ratio"]) * df["max_cycle"]
df["rul_ratio"] = df["remaining_cycles"] / df["max_cycle"]
df["cycle_norm"] = df["cycle"] / df["max_cycle"]
print(f"\n[DATA] Rows: {len(df):,} | Batteries: {df['battery_id'].nunique()}")
print(f"       Types: {df['battery_type'].unique().tolist()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[FEATURES] Engineering features...")

# Cumulative voltage drop per battery — tracks total degradation
# accumulated so far without revealing the endpoint (max_cycle).
df["cumul_voltage_drop"] = df.groupby("battery_id")["voltage_drop"].cumsum()

# Percentile rank of cycle within each battery's observed history.
# Gives a bounded relative position [0,1] without needing max_cycle.
# cycle_ratio = cycle/max_cycle is intentionally excluded because
# it leaks max_cycle (the total lifespan) directly into the input.
df["cycle_rank"] = df.groupby("battery_id")["cycle"].rank(pct=True)

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
    "cycle_norm",          # normalized lifecycle position [0,1]
]

# Rolling mean and std per battery.
# Window=10 is half of SEQ_LEN=20, giving the LSTM a smoothed
# recent-history signal alongside the raw per-cycle values.
ROLLING_WINDOW = 10
print(f"[FEATURES] Rolling mean/std (window={ROLLING_WINDOW})...")
rolling_feats = []
for feat in PHYSICAL_FEATURES:
    df[f"{feat}_mean"] = df.groupby("battery_id")[feat].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    )
    df[f"{feat}_std"] = df.groupby("battery_id")[feat].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std()
    ).fillna(0)
    rolling_feats.extend([f"{feat}_mean", f"{feat}_std"])

ALL_FEATURES = PHYSICAL_FEATURES + rolling_feats
SOH_FEATURES = ALL_FEATURES
RUL_FEATURES = ALL_FEATURES
SOH_TARGET   = "SoH_smoothed"
RUL_TARGET   = "rul_ratio"

print(f"[FEATURES] Total input features: {len(ALL_FEATURES)}")

# ─────────────────────────────────────────────
# 3. SoH LABEL SMOOTHING (per battery)
# Per-cycle SoH measurements contain noise that caused the extremely
# jagged predictions visible in v3 plots. Savitzky-Golay smoothing
# (window=11, polyorder=2) preserves the monotonic degradation trend
# while removing high-frequency noise.
# Raw SoH column is kept intact for actual EOL threshold detection.
# ─────────────────────────────────────────────
print("[FEATURES] Smoothing SoH labels per battery (Savitzky-Golay)...")
smoothed_parts = []
for _, grp in df.groupby("battery_id"):
    grp  = grp.sort_values("cycle")
    vals = grp["SoH"].values
    smoothed = (
        np.clip(savgol_filter(vals, window_length=11, polyorder=2), 0.0, 1.0)
        if len(vals) >= 11 else vals
    )
    smoothed_parts.append(pd.Series(smoothed, index=grp.index))
df["SoH_smoothed"] = pd.concat(smoothed_parts)

# ─────────────────────────────────────────────
# 4. CORRELATION MATRICES
# ─────────────────────────────────────────────
corr_feats = [
    "avg_voltage", "min_voltage", "avg_current", "duration",
    "voltage_drop", "temp_change", "cycle",
    "cumul_voltage_drop", "cycle_rank"
]
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(df[corr_feats + [SOH_TARGET]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0], linewidths=0.5)
axes[0].set_title("Correlation Matrix — SoH Features", fontsize=12, fontweight="bold")

sns.heatmap(df[corr_feats + [RUL_TARGET]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1], linewidths=0.5)
axes[1].set_title("Correlation Matrix — RUL Features", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_correlation_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("[PLOT] Saved: 01_correlation_matrices.png")

# ─────────────────────────────────────────────
# 5. STRATIFIED BATTERY-GROUP SPLIT  70/20/10
# Stratified by battery_type ensures all three chemistries
# (regular, recommissioned, second_life) appear in every split.
# More validation batteries (20%) improves val_loss stability.
# ─────────────────────────────────────────────
df_sorted = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
rng = np.random.default_rng(SEED)

train_bats, val_bats, test_bats = [], [], []
for _, grp in df.groupby("battery_type"):
    bats = np.array(sorted(grp["battery_id"].unique()))
    rng.shuffle(bats)
    n    = len(bats)
    n_tr = max(1, int(n * 0.70))
    n_vl = max(1, int(n * 0.20))
    n_te = max(1, n - n_tr - n_vl)
    if n_tr + n_vl + n_te > n:
        n_tr -= 1
    train_bats.extend(bats[:n_tr].tolist())
    val_bats.extend(bats[n_tr:n_tr + n_vl].tolist())
    test_bats.extend(bats[n_tr + n_vl:n_tr + n_vl + n_te].tolist())

df_train = df_sorted[df_sorted["battery_id"].isin(train_bats)].reset_index(drop=True)
df_val   = df_sorted[df_sorted["battery_id"].isin(val_bats)].reset_index(drop=True)
df_test  = df_sorted[df_sorted["battery_id"].isin(test_bats)].reset_index(drop=True)

print(f"\n[SPLIT — Stratified 70/20/10]")
print(f"  Train : {len(df_train):>6,} rows | {len(train_bats)} batteries: {sorted(train_bats)}")
print(f"  Val   : {len(df_val):>6,} rows | {len(val_bats)} batteries:   {sorted(val_bats)}")
print(f"  Test  : {len(df_test):>6,} rows | {len(test_bats)} batteries:  {sorted(test_bats)}")
print(f"  Train types: {sorted(df_train['battery_type'].unique().tolist())}")
print(f"  Val   types: {sorted(df_val['battery_type'].unique().tolist())}")
print(f"  Test  types: {sorted(df_test['battery_type'].unique().tolist())}")

# ─────────────────────────────────────────────
# 6. SCALING
# All scalers fitted on train only — no data leakage.
# SoH uses no target scaler: sigmoid output is already [0,1].
# RUL target is MinMaxScaled; Huber delta=0.05 is calibrated
# for this [0,1] scaled space (≈50 cycles in original scale).
# ─────────────────────────────────────────────
soh_feat_scaler = MinMaxScaler()
rul_feat_scaler = MinMaxScaler()
rul_tgt_scaler  = MinMaxScaler()

X_tr_soh  = soh_feat_scaler.fit_transform(df_train[SOH_FEATURES])
y_tr_soh  = df_train[SOH_TARGET].values.reshape(-1, 1)
X_val_soh = soh_feat_scaler.transform(df_val[SOH_FEATURES])
y_val_soh = df_val[SOH_TARGET].values.reshape(-1, 1)
X_te_soh  = soh_feat_scaler.transform(df_test[SOH_FEATURES])
y_te_soh  = df_test[SOH_TARGET].values.reshape(-1, 1)

X_tr_rul  = rul_feat_scaler.fit_transform(df_train[RUL_FEATURES])
y_tr_rul  = rul_tgt_scaler.fit_transform(df_train[[RUL_TARGET]])
X_val_rul = rul_feat_scaler.transform(df_val[RUL_FEATURES])
y_val_rul = rul_tgt_scaler.transform(df_val[[RUL_TARGET]])
X_te_rul  = rul_feat_scaler.transform(df_test[RUL_FEATURES])
y_te_rul  = rul_tgt_scaler.transform(df_test[[RUL_TARGET]])

print("\n[SCALE] Scalers fitted on train only. No data leakage.")

# ─────────────────────────────────────────────
# 7. BATTERY-WISE SEQUENCE BUILDER
# Sliding-window sequences never cross battery boundaries,
# preventing temporal leakage between different lifecycles.
# ─────────────────────────────────────────────
def build_sequences_by_battery(df_subset, scaled_X, scaled_y, seq_len):
    X_out, y_out = [], []
    for _, grp in df_subset.reset_index(drop=True).groupby("battery_id"):
        indices = grp.sort_values("cycle").index.values
        for i in range(len(indices) - seq_len):
            X_out.append(scaled_X[indices[i: i + seq_len]])
            y_out.append(scaled_y[indices[i + seq_len]])
    return np.array(X_out), np.array(y_out)

print(f"\n[SEQ] Building sequences (seq_len={SEQ_LEN}, no boundary leakage)...")
Xtr_soh, ytr_soh = build_sequences_by_battery(df_train, X_tr_soh, y_tr_soh, SEQ_LEN)
Xvl_soh, yvl_soh = build_sequences_by_battery(df_val,   X_val_soh, y_val_soh, SEQ_LEN)
Xte_soh, yte_soh = build_sequences_by_battery(df_test,  X_te_soh,  y_te_soh,  SEQ_LEN)

Xtr_rul, ytr_rul = build_sequences_by_battery(df_train, X_tr_rul, y_tr_rul, SEQ_LEN)
Xvl_rul, yvl_rul = build_sequences_by_battery(df_val,   X_val_rul, y_val_rul, SEQ_LEN)
Xte_rul, yte_rul = build_sequences_by_battery(df_test,  X_te_rul,  y_te_rul,  SEQ_LEN)

print(f"[SEQ] SoH — Train:{Xtr_soh.shape} | Val:{Xvl_soh.shape} | Test:{Xte_soh.shape}")
print(f"[SEQ] RUL — Train:{Xtr_rul.shape} | Val:{Xvl_rul.shape} | Test:{Xte_rul.shape}")

# Sample weights for RUL: early cycles are more important
# Build weights from cycle_norm values aligned with sequences
def build_sample_weights(df_subset, seq_len):
    weights = []
    for _, grp in df_subset.reset_index(drop=True).groupby("battery_id"):
        indices = grp.sort_values("cycle").index.values
        for i in range(len(indices) - seq_len):
            cn = df_subset.loc[indices[i + seq_len], "cycle_norm"]
            weights.append(1 + (1 - cn))
    return np.array(weights)

rul_train_weights = build_sample_weights(df_train, SEQ_LEN)
print(f"[WEIGHTS] RUL sample weights: min={rul_train_weights.min():.2f}, max={rul_train_weights.max():.2f}")

n_feat = len(ALL_FEATURES)

# ─────────────────────────────────────────────
# 8. MODEL BUILDERS
# ─────────────────────────────────────────────
def build_soh_model(seq_len, n_feat):
    """
    3-layer LSTM with BatchNorm + Dropout(0.15) after each layer.
    Sigmoid output: constrains predictions to [0,1] matching SoH range.
    Units 128→64→32: smaller model generalises better with ~16 train batteries.
    Dropout 0.15 (was 0.2 in v3): BatchNorm provides implicit regularisation
    so aggressive dropout is not needed.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_feat)),
        BatchNormalization(),
        Dropout(0.15),

        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.15),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.15),

        Dense(64, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.1),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def build_rul_model(seq_len, n_feat):
    """
    2-layer LSTM (reduced from 3) with BatchNorm + Dropout(0.20).
    Smaller model (64→32 units) reduces overfitting with limited batteries.
    Huber(delta=0.05) calibrated for [0,1] scaled rul_ratio target.
    ReLU output ensures non-negative predictions.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_feat)),
        BatchNormalization(),
        Dropout(0.20),

        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.20),

        Dense(32, activation="relu"),
        Dense(1,  activation="relu"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model


def make_callbacks(model_name):
    return [
        EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            f"{MODEL_DIR}/{model_name}_best.keras",
            monitor="val_loss", save_best_only=True, verbose=0
        ),
        # patience=8 (was 5 in v3): LR was halved 4 times in v3 SoH model,
        # collapsing learning too early. More patience lets loss recover first.
        # min_lr=1e-6 (was 1e-5): finer lower bound for late-stage fine-tuning.
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1
        ),
    ]

# ─────────────────────────────────────────────
# 9. TRAIN SoH MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training SoH Model")
print("=" * 60)

soh_model = build_soh_model(SEQ_LEN, n_feat)
soh_model.summary()

soh_history = soh_model.fit(
    Xtr_soh, ytr_soh,
    validation_data=(Xvl_soh, yvl_soh),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=make_callbacks("soh"),
    verbose=1
)
print("[TRAIN] SoH model training complete.")

# ─────────────────────────────────────────────
# 10. TRAIN RUL MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Training RUL Model")
print("=" * 60)

rul_model = build_rul_model(SEQ_LEN, n_feat)
rul_model.summary()

rul_history = rul_model.fit(
    Xtr_rul, ytr_rul,
    validation_data=(Xvl_rul, yvl_rul),
    epochs=EPOCHS, batch_size=BATCH,
    sample_weight=rul_train_weights,
    callbacks=make_callbacks("rul"),
    verbose=1
)
print("[TRAIN] RUL model training complete.")

# ─────────────────────────────────────────────
# 10b. VALIDATION EVALUATION LOGGING
# ─────────────────────────────────────────────
print("\n[VAL EVAL] Evaluating models on validation set...")
soh_val_results = soh_model.evaluate(Xvl_soh, yvl_soh, verbose=0)
print(f"  SoH Val — Loss: {soh_val_results[0]:.6f} | RMSE: {soh_val_results[1]:.6f}")
rul_val_results = rul_model.evaluate(Xvl_rul, yvl_rul, verbose=0)
print(f"  RUL Val — Loss: {rul_val_results[0]:.6f} | RMSE: {rul_val_results[1]:.6f}")

# ─────────────────────────────────────────────
# 11. SAVE MODELS & SCALERS
# ─────────────────────────────────────────────
soh_model.save(f"{MODEL_DIR}/soh_model.keras")
rul_model.save(f"{MODEL_DIR}/rul_model.keras")
joblib.dump(soh_feat_scaler, f"{SCALER_DIR}/soh_feat_scaler.pkl")
joblib.dump(rul_feat_scaler, f"{SCALER_DIR}/rul_feat_scaler.pkl")
joblib.dump(rul_tgt_scaler,  f"{SCALER_DIR}/rul_tgt_scaler.pkl")
print(f"\n[SAVE] Models  → {MODEL_DIR}/")
print(f"[SAVE] Scalers → {SCALER_DIR}/")

# ─────────────────────────────────────────────
# 12. PLOT HELPERS
# ─────────────────────────────────────────────
def plot_training(history, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes[0].plot(history.history["loss"],     label="Train Loss",  color="steelblue")
    axes[0].plot(history.history["val_loss"], label="Val Loss",    color="orangered", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["rmse"],     label="Train RMSE",  color="steelblue")
    axes[1].plot(history.history["val_rmse"], label="Val RMSE",    color="orangered", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE Convergence"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_actual_vs_predicted(true, pred, title, ylabel, out_path, threshold=None):
    plt.figure(figsize=(14, 5))
    plt.plot(true, label="Actual",    color="steelblue", linewidth=1.2)
    plt.plot(pred, label="Predicted", color="orangered", linewidth=1.2, linestyle="--")
    if threshold is not None:
        plt.axhline(threshold, color="black", linestyle=":", linewidth=1.5,
                    label=f"Threshold ({threshold})")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Test Sample Index"); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


plot_training(soh_history, "SoH Model — Training Progress",
              f"{OUT_DIR}/02_soh_training_progress.png")
print("[PLOT] Saved: 02_soh_training_progress.png")

plot_training(rul_history, "RUL Model — Training Progress",
              f"{OUT_DIR}/03_rul_training_progress.png")
print("[PLOT] Saved: 03_rul_training_progress.png")

# ─────────────────────────────────────────────
# 13. SoH EVALUATION
# ─────────────────────────────────────────────
soh_pred     = soh_model.predict(Xte_soh, verbose=0).flatten()
soh_true     = yte_soh.flatten()
soh_rmse_pct = np.sqrt(mean_squared_error(soh_true, soh_pred)) * 100
soh_mae_pct  = mean_absolute_error(soh_true, soh_pred) * 100
soh_r2       = r2_score(soh_true, soh_pred)

print(f"\n[SoH RESULTS]")
print(f"  Test RMSE : {soh_rmse_pct:.2f}%")
print(f"  Test MAE  : {soh_mae_pct:.2f}%")
print(f"  Test R²   : {soh_r2:.4f}")

plot_actual_vs_predicted(
    soh_true, soh_pred,
    title=f"SoH Prediction vs Actual  |  RMSE={soh_rmse_pct:.2f}%  MAE={soh_mae_pct:.2f}%  R²={soh_r2:.4f}",
    ylabel="State of Health (SoH)",
    out_path=f"{OUT_DIR}/04_soh_prediction.png",
    threshold=FAILURE_THRESHOLD
)
print("[PLOT] Saved: 04_soh_prediction.png")

# Save SoH predictions CSV
pd.DataFrame({"actual_soh": soh_true, "predicted_soh": soh_pred}).to_csv(
    f"{OUT_DIR}/soh_predictions.csv", index=False
)
print("[SAVE] soh_predictions.csv")

# ─────────────────────────────────────────────
# 14. PER-BATTERY SoH SUBPLOT GRID (Test Set)
# ─────────────────────────────────────────────
test_bat_ids = sorted(df_test["battery_id"].unique())
ncols = min(3, len(test_bat_ids))
nrows = math.ceil(len(test_bat_ids) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
df_te_reset = df_test.reset_index(drop=True)

for idx, bat_id in enumerate(test_bat_ids):
    r, c    = divmod(idx, ncols)
    ax      = axes[r][c]
    grp     = df_te_reset[df_te_reset["battery_id"] == bat_id].sort_values("cycle")
    indices = grp.index.values

    if len(indices) <= SEQ_LEN:
        ax.set_title(f"{bat_id} (too few cycles)"); ax.axis("off"); continue

    X_bat  = np.array([X_te_soh[indices[i:i + SEQ_LEN]] for i in range(len(indices) - SEQ_LEN)])
    preds  = soh_model.predict(X_bat, verbose=0).flatten()
    trues  = y_te_soh[indices[SEQ_LEN:]].flatten()
    cycles = grp.loc[indices[SEQ_LEN:], "cycle"].values

    ax.plot(cycles, trues, label="Actual",    color="steelblue", linewidth=1.2)
    ax.plot(cycles, preds, label="Predicted", color="orangered", linewidth=1.2, linestyle="--")
    ax.axhline(FAILURE_THRESHOLD, color="black", linestyle=":", linewidth=1)
    ax.set_title(bat_id, fontweight="bold")
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

for i in range(len(test_bat_ids), nrows * ncols):
    axes[i // ncols][i % ncols].axis("off")

plt.suptitle("Per-Battery SoH Degradation — Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_soh_per_battery.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("[PLOT] Saved: 05_soh_per_battery.png")

# ─────────────────────────────────────────────
# 15. RUL EVALUATION
# Target is rul_ratio — inverse_transform gives ratio, multiply by max_cycle
# to convert back to actual cycle counts for interpretable metrics.
# ─────────────────────────────────────────────
# Get predicted ratio and true ratio (in original rul_ratio scale)
rul_pred_ratio = rul_tgt_scaler.inverse_transform(
    rul_model.predict(Xte_rul, verbose=0)
).flatten()
rul_pred_ratio = np.clip(rul_pred_ratio, 0, None)
rul_true_ratio = rul_tgt_scaler.inverse_transform(yte_rul).flatten()

# Get max_cycle for each test sequence to convert ratio → cycles
test_max_cycles = []
for _, grp in df_test.reset_index(drop=True).groupby("battery_id"):
    indices = grp.sort_values("cycle").index.values
    for i in range(len(indices) - SEQ_LEN):
        test_max_cycles.append(df_test.loc[indices[i + SEQ_LEN], "max_cycle"])
test_max_cycles = np.array(test_max_cycles)

rul_pred = rul_pred_ratio * test_max_cycles
rul_true = rul_true_ratio * test_max_cycles

rul_rmse = np.sqrt(mean_squared_error(rul_true, rul_pred))
rul_mae  = mean_absolute_error(rul_true, rul_pred)
rul_r2   = r2_score(rul_true, rul_pred)

print(f"\n[RUL RESULTS]")
print(f"  Test RMSE : {rul_rmse:.2f} cycles")
print(f"  Test MAE  : {rul_mae:.2f} cycles")
print(f"  Test R²   : {rul_r2:.4f}")

plot_actual_vs_predicted(
    rul_true, rul_pred,
    title=f"RUL Prediction vs Actual  |  RMSE={rul_rmse:.2f} cycles  MAE={rul_mae:.2f} cycles  R²={rul_r2:.4f}",
    ylabel="Remaining Useful Life (cycles)",
    out_path=f"{OUT_DIR}/06_rul_prediction.png"
)
print("[PLOT] Saved: 06_rul_prediction.png")

# Save RUL predictions CSV
pd.DataFrame({"actual_rul_cycles": rul_true, "predicted_rul_cycles": rul_pred}).to_csv(
    f"{OUT_DIR}/rul_predictions.csv", index=False
)
print("[SAVE] rul_predictions.csv")

# ─────────────────────────────────────────────
# 16. SCATTER PLOTS — Actual vs Predicted
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, true, pred, xlabel, ylabel, title, color in [
    (axes[0], soh_true, soh_pred,
     "Actual SoH", "Predicted SoH",
     f"SoH: Actual vs Predicted\nRMSE={soh_rmse_pct:.2f}%  R²={soh_r2:.4f}",
     "steelblue"),
    (axes[1], rul_true, rul_pred,
     "Actual RUL (cycles)", "Predicted RUL (cycles)",
     f"RUL: Actual vs Predicted\nRMSE={rul_rmse:.2f} cycles  R²={rul_r2:.4f}",
     "darkorange"),
]:
    ax.scatter(true, pred, alpha=0.3, s=10, color=color)
    lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_scatter_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("[PLOT] Saved: 07_scatter_actual_vs_predicted.png")

# ─────────────────────────────────────────────
# 17. PER-BATTERY EOL CYCLE PREDICTION
# Uses predicted SoH to find when each battery crosses FAILURE_THRESHOLD.
# Raw SoH column used for actual EOL to preserve true threshold crossing.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Per-Battery End-of-Life Cycle Prediction")
print("=" * 60)

soh_all_feat = soh_feat_scaler.transform(df_sorted[SOH_FEATURES])
soh_all_tgt  = df_sorted[SOH_TARGET].values.reshape(-1, 1)
Xall_soh, _  = build_sequences_by_battery(df_sorted, soh_all_feat, soh_all_tgt, SEQ_LEN)
soh_all_pred = soh_model.predict(Xall_soh, verbose=0).flatten()

df_ev_parts = []
for _, grp in df_sorted.groupby("battery_id"):
    df_ev_parts.append(grp.sort_values("cycle").iloc[SEQ_LEN:])
df_ev = pd.concat(df_ev_parts).reset_index(drop=True)
df_ev["predicted_SoH"] = soh_all_pred

results = []
for bat_id, grp in df_ev.groupby("battery_id"):
    grp          = grp.sort_values("cycle")
    actual_eol   = grp.loc[grp["SoH"] < FAILURE_THRESHOLD, "cycle"]
    pred_eol     = grp.loc[grp["predicted_SoH"] < FAILURE_THRESHOLD, "cycle"]
    actual_cycle = int(actual_eol.iloc[0]) if len(actual_eol) > 0 else None
    pred_cycle   = int(pred_eol.iloc[0])   if len(pred_eol)   > 0 else None
    error        = (pred_cycle - actual_cycle) if (actual_cycle and pred_cycle) else None
    abs_error    = abs(error) if error is not None else None

    results.append({
        "Battery":             bat_id,
        "Actual EOL Cycle":    actual_cycle,
        "Predicted EOL Cycle": pred_cycle,
        "Error (cycles)":      error,
        "Abs Error (cycles)":  abs_error,
    })
    print(f"  {bat_id:15s}  |  Actual:{str(actual_cycle):>6}  "
          f"|  Predicted:{str(pred_cycle):>6}  |  Error:{str(error):>5} cycles")

pd.DataFrame(results).to_csv(f"{OUT_DIR}/08_rul_eol_results.csv", index=False)
print("[SAVE] 08_rul_eol_results.csv")

# ─────────────────────────────────────────────
# 18. METRICS JSON + FINAL SUMMARY
# ─────────────────────────────────────────────
best_soh_epoch     = int(np.argmin(soh_history.history["val_loss"])) + 1
best_rul_epoch     = int(np.argmin(rul_history.history["val_loss"])) + 1
valid_abs_errors   = [r["Abs Error (cycles)"] for r in results if r["Abs Error (cycles)"] is not None]
eol_mean_abs_error = float(np.mean(valid_abs_errors)) if valid_abs_errors else None

metrics = {
    "soh_rmse_pct":              round(float(soh_rmse_pct), 4),
    "soh_mae_pct":               round(float(soh_mae_pct),  4),
    "soh_r2":                    round(float(soh_r2),       4),
    "rul_rmse_cycles":           round(float(rul_rmse),     4),
    "rul_mae_cycles":            round(float(rul_mae),      4),
    "rul_r2":                    round(float(rul_r2),       4),
    "eol_mean_abs_error_cycles": round(eol_mean_abs_error,  1) if eol_mean_abs_error else None,
    "best_soh_epoch":            best_soh_epoch,
    "best_rul_epoch":            best_rul_epoch,
    "train_batteries":           sorted(train_bats),
    "val_batteries":             sorted(val_bats),
    "test_batteries":            sorted(test_bats),
    "seq_len":                   SEQ_LEN,
    "batch_size":                BATCH,
    "epochs_ceiling":            EPOCHS,
    "patience":                  PATIENCE,
    "failure_threshold":         FAILURE_THRESHOLD,
    "features":                  ALL_FEATURES,
    "soh_label_smoothing":       "Savitzky-Golay window=11 polyorder=2",
    "rul_huber_delta_scaled":    0.05,
    "split_ratios":              "70/20/10 stratified by battery_type",
    "rul_target":                "rul_ratio (normalized)",
    "soh_loss":                  "Huber(delta=0.1)",
}

with open(f"{OUT_DIR}/09_metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[SAVE] 09_metrics_summary.json")

print("\n" + "=" * 60)
print("  FINAL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"  SoH Model  →  RMSE : {soh_rmse_pct:.2f}%  |  MAE : {soh_mae_pct:.2f}%  |  R² : {soh_r2:.4f}")
print(f"  RUL Model  →  RMSE : {rul_rmse:.2f} cycles  |  MAE : {rul_mae:.2f} cycles  |  R² : {rul_r2:.4f}")
if eol_mean_abs_error:
    print(f"  EOL Pred   →  Mean |Error| : {eol_mean_abs_error:.1f} cycles")
print(f"  Best Epochs → SoH: {best_soh_epoch}  |  RUL: {best_rul_epoch}")
print("=" * 60)
print(f"\n[DONE] Outputs  → {OUT_DIR}")
print(f"[DONE] Models   → {MODEL_DIR}")
print(f"[DONE] Scalers  → {SCALER_DIR}")

print("""
[OUTPUT FILES]
  01_correlation_matrices.png        — feature correlations with targets
  02_soh_training_progress.png       — SoH loss + RMSE convergence
  03_rul_training_progress.png       — RUL Huber loss + RMSE convergence
  04_soh_prediction.png              — SoH actual vs predicted (test set)
  05_soh_per_battery.png             — per-battery SoH degradation curves
  06_rul_prediction.png              — RUL actual vs predicted (test set)
  07_scatter_actual_vs_predicted.png — dual scatter with R² in titles
  08_rul_eol_results.csv             — per-battery EOL + Abs Error
  09_metrics_summary.json            — complete metrics + config
  soh_predictions.csv                — SoH actual vs predicted values
  rul_predictions.csv                — RUL actual vs predicted values (cycles)
""")
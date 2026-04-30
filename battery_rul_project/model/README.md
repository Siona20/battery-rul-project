# Battery SoH & RUL Prediction Models

This directory contains the implementation of LSTM-based deep learning models designed to predict the **State of Health (SoH)** and **Remaining Useful Life (RUL)** of lithium-ion batteries.

## 📂 Folder Structure

- `lstm_model.py`: The core pipeline (v5) for training, evaluation, and visualization.
- `outputs/`:
  - Visualizations (Correlation matrices, Training progress, Prediction plots, Scatter plots).
  - `soh_predictions.csv`: SoH actual vs predicted values.
  - `rul_predictions.csv`: RUL actual vs predicted values (in cycles).
  - `08_rul_eol_results.csv`: Per-battery End-of-Life prediction errors.
  - `09_metrics_summary.json`: Comprehensive performance metrics and config.
- `outputs/models/`: Saved TensorFlow/Keras models (`.keras` format) for both SoH and RUL.
- `outputs/scalers/`: Saved `MinMaxScaler` objects (`.pkl`) used for feature normalization.

## 🚀 Key Features

### 1. Dual-Model Architecture
- **SoH Model**: A 3-layer LSTM (128→64→32 units) with BatchNormalization and Dropout(0.15). Sigmoid output constrains predictions to [0, 1]. Loss: Huber(delta=0.1).
- **RUL Model**: A simplified 2-layer LSTM (64→32 units) with BatchNormalization and Dropout(0.20). ReLU output ensures non-negative predictions. Loss: Huber(delta=0.05).

### 2. Feature Engineering
- **10 Physical Features**: avg_voltage, min_voltage, avg_current, duration, voltage_drop, temp_change, cycle, cumul_voltage_drop, cycle_rank, cycle_norm.
- **Rolling Statistics**: Rolling mean and std (window=10) for each physical feature → 30 total input features.
- **Lifecycle Anchor**: `cycle_norm = cycle / max_cycle` helps the model understand battery lifecycle position.
- **SoH Label Smoothing**: Savitzky-Golay filter (window=11, polyorder=2) removes noise while preserving degradation trend.

### 3. Normalized RUL Target
- **Target**: `rul_ratio = remaining_cycles / max_cycle` instead of raw `remaining_cycles`.
- This normalizes all batteries to the same [0, 1] scale, eliminating scale imbalance across batteries with different total lifespans.
- During evaluation, predictions are converted back to cycles: `predicted_cycles = predicted_ratio × max_cycle`.

### 4. Robust Data Strategy
- **Battery-Group Split (70/20/10)**: Train=70%, Validation=20%, Test=10% — split by battery identity, stratified by battery_type. More validation batteries (20%) improves val_loss stability.
- **Sequence Building**: Sliding window sequences (length=20) are built strictly within individual battery lifecycles to prevent data leakage across boundaries.
- **Scalers fitted on train only**: No data leakage from validation or test sets.

### 5. Sample Weighting (RUL)
- Early-life cycles are weighted higher: `weight = 1 + (1 - cycle_norm)`.
- This improves early-life RUL prediction accuracy, which is more valuable for real-world applications.

### 6. Advanced Training Logic
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-6) — automatically reduces learning rate when validation loss plateaus.
- **Early Stopping**: Patience=20 with best weight restoration to ensure maximum generalization.
- **Validation Evaluation Logging**: Both models are explicitly evaluated on the validation set after training, with loss and RMSE printed.

## 📊 Evaluation Metrics

The models are evaluated using:
- **RMSE (Root Mean Squared Error)**: To penalize large prediction gaps.
- **MAE (Mean Absolute Error)**: To measure average prediction accuracy.
- **R² Score**: To measure the variance explained by the model.
- **EOL Cycle Error**: Direct measurement of how many cycles off the "End-of-Life" prediction was.

## 📁 Output Files

| File | Description |
|------|-------------|
| `01_correlation_matrices.png` | Feature correlations with SoH and RUL targets |
| `02_soh_training_progress.png` | SoH loss + RMSE convergence curves |
| `03_rul_training_progress.png` | RUL Huber loss + RMSE convergence curves |
| `04_soh_prediction.png` | SoH actual vs predicted (test set) |
| `05_soh_per_battery.png` | Per-battery SoH degradation curves |
| `06_rul_prediction.png` | RUL actual vs predicted (test set) |
| `07_scatter_actual_vs_predicted.png` | Dual scatter plots with R² |
| `08_rul_eol_results.csv` | Per-battery EOL cycle prediction errors |
| `09_metrics_summary.json` | Complete metrics + config summary |
| `soh_predictions.csv` | SoH actual vs predicted values |
| `rul_predictions.csv` | RUL actual vs predicted values (cycles) |

## 🛠️ How to Run

1. Ensure your processed data is available at `../processed_data/final_dataset_with_soh.csv`.
2. Run the training pipeline:
   ```bash
   python lstm_model.py
   ```
3. Check the `outputs/` folder for generated graphs, prediction CSVs, and `09_metrics_summary.json` for the final performance report.

## 🔮 Future Improvements (Planned)

- **Attention Mechanism**: Add attention layer on top of LSTM for interpretability.
- **Multi-Task Learning**: Combine SoH + RUL into a single shared-encoder model.
- **Cross-Validation**: Use GroupKFold for more robust evaluation across battery groups.

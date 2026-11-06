"""
Experiment 1 — Standard Temporal Evaluation
4 models (LR, RF, LightGBM, XGBoost) x 2 targets (PM2.5, PM10)
Reduced feature set: PM10/PM2.5 + WindSpeed + Humidity
Protocol per plan v8 / FINAL §6:
  - Min-Max normalization fit on TRAIN ONLY, applied to test
  - Batch predict() timing on full test set
  - Model size = serialized file size (KB)
  - Hyperparameters: RF = 100 trees / depth 8 (continuity with original paper)
    LightGBM/XGBoost = stated library defaults, no tuning
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# Assumes this script sits in the same folder as the CSV files.
# Adjust these paths if you keep your data elsewhere.
DATA_DIR = Path(".")
OUT_DIR = Path("./experiment1_out")
OUT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "reduced_2024_train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "reduced_2024_test.csv", parse_dates=["Date"])

# ---- Target/feature definitions (reduced feature set) ----
TARGETS = {
    "PM2.5": ["PM10", "WindSpeed", "Humidity"],
    "PM10": ["PM2.5", "WindSpeed", "Humidity"],
}

# ---- Fixed hyperparameters (documented, no test-set tuning) ----
def make_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1),
        "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
    }

results = []

for target_name, feature_cols in TARGETS.items():
    X_train_raw = train[feature_cols].values
    y_train = train[target_name].values
    X_test_raw = test[feature_cols].values
    y_test = test[target_name].values

    # Normalization: fit on TRAIN ONLY (protocol requirement)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    models = make_models()

    for model_name, model in models.items():
        # --- Train ---
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # --- Predict (batch, full test set) ---
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        # --- Metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # --- Model size (serialized, KB) ---
        model_path = OUT_DIR / f"{target_name}_{model_name}.joblib"
        joblib.dump(model, model_path)
        size_kb = model_path.stat().st_size / 1024

        results.append({
            "Target": target_name,
            "Model": model_name,
            "Features": ", ".join(feature_cols),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Train_Time_sec": round(train_time, 4),
            "Predict_Time_ms": round(predict_time * 1000, 4),
            "Model_Size_KB": round(size_kb, 2),
        })

        print(f"[{target_name}] {model_name:16s} | MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f} "
              f"| train={train_time:.3f}s predict={predict_time*1000:.3f}ms size={size_kb:.1f}KB")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "experiment1_results.csv", index=False)

print("\n=== Experiment 1 Results ===")
print(results_df.to_string(index=False))

# Save scalers used (needed for reproducibility / later experiments)
print(f"\nSaved results to {OUT_DIR / 'experiment1_results.csv'}")
print(f"Saved {len(results)} trained model files to {OUT_DIR}")

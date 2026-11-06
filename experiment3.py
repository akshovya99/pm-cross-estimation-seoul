"""
Experiment 3 — Spatial Generalization (Full Leave-One-Station-Out CV)

Tests whether models generalize to UNSEEN stations, not just later dates.
Directly answers the original paper's stated limitation: "trained and
validated exclusively on Seoul's 2024 data... limits generalizability."

Design (per project plan §6-7):
  - 50 folds, one per station. In each fold:
      - Training stations = the other 49 stations, Jan 8-Oct 14 data only
        (same temporal window as Experiment 1's train split)
      - Held-out station = Oct 15-Dec 31 data only (test period)
    This preserves BOTH the temporal split and the spatial split inside a
    single experiment, and is NOT directly comparable to Experiment 1
    (which trains on all 50 stations) — see note in results.
  - Min-Max scaler fit on the fold's TRAINING data only (49 stations,
    train-period) — refit fresh for every fold. No information from the
    held-out station enters preprocessing.
  - NOTE ON IMPUTATION: the source dataset provided to this pipeline has
    already been through LOCF imputation upstream (verified structurally:
    the file is station-major and date-sorted per station, consistent with
    correct per-station imputation). Because the pre-imputation raw data
    (with missing values) is not available to this pipeline, per-fold
    re-imputation is not performed here — only per-fold scaling is refit.
    This is a limitation to state plainly in the Methodology.
  - Stochastic models (RF, LightGBM, XGBoost) run across multiple seeds;
    Linear Regression is deterministic and run once per fold.
  - Metrics: MAE, RMSE, R2 per fold. Reported as mean +/- SD and median + IQR
    across the 50 stations. MAE/RMSE are the LEAD metrics; R2 is reported
    second, with a caveat, since per-station R2 can be unstable when a
    held-out station has low test-window variance.

Runtime note: ~50 folds x 2 targets x (1 LR + 3 stochastic models x N_SEEDS)
fits. With N_SEEDS=5 that's 50 x 2 x 16 = 1600 model fits. Given Experiment 1's
sub-second training times, this should complete in well under 30 minutes on a
normal laptop. Reduce N_SEEDS to lower runtime if needed (5-10 is the
documented acceptable range).
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")  # suppress sklearn "no feature names" noise

DATA_DIR = Path(".")
OUT_DIR = Path("./experiment3_out")
OUT_DIR.mkdir(exist_ok=True)

N_SEEDS = 5  # documented acceptable range: 5-10. Increase to 10 if runtime allows.
SEEDS = list(range(N_SEEDS))

TRAIN_END = "2024-10-14"
TEST_START = "2024-10-15"

TARGETS = {
    "PM2.5": ["PM10", "WindSpeed", "Humidity"],
    "PM10": ["PM2.5", "WindSpeed", "Humidity"],
}

full = pd.read_csv(DATA_DIR / "reduced_2024_full.csv", parse_dates=["Date"])
stations = sorted(full["Station"].unique())
print(f"Loaded {len(full)} rows, {len(stations)} stations.")

def make_model(model_name, seed):
    if model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=seed
        )
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(random_state=seed, verbose=-1)
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(random_state=seed, verbosity=0)

MODEL_NAMES = ["LinearRegression", "RandomForest", "LightGBM", "XGBoost"]
DETERMINISTIC = {"LinearRegression"}

results = []
t_start = time.time()
fold_count = 0

for held_out_station in stations:
    fold_count += 1
    train_stations_data = full[
        (full["Station"] != held_out_station) & (full["Date"] <= TRAIN_END)
    ]
    test_station_data = full[
        (full["Station"] == held_out_station) & (full["Date"] >= TEST_START)
    ]

    for target_name, feature_cols in TARGETS.items():
        X_train_raw = train_stations_data[feature_cols].values
        y_train = train_stations_data[target_name].values
        X_test_raw = test_station_data[feature_cols].values
        y_test = test_station_data[target_name].values

        # Refit scaler on THIS FOLD's training data only (49 stations, train period)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for model_name in MODEL_NAMES:
            seeds_to_run = [None] if model_name in DETERMINISTIC else SEEDS

            for seed in seeds_to_run:
                model = make_model(model_name, seed if seed is not None else 0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                results.append({
                    "HeldOutStation": held_out_station,
                    "Target": target_name,
                    "Model": model_name,
                    "Seed": seed if seed is not None else "N/A",
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                })

    if fold_count % 10 == 0 or fold_count == len(stations):
        elapsed = time.time() - t_start
        print(f"  ...completed {fold_count}/{len(stations)} folds ({elapsed:.1f}s elapsed)")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "experiment3_loso_raw.csv", index=False)
print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
print(f"Saved raw per-fold, per-seed results to {OUT_DIR / 'experiment3_loso_raw.csv'} "
      f"({len(results_df)} rows)")

# ---- Aggregate: mean/SD and median/IQR across stations, per model/target ----
# First average across seeds within each station (for stochastic models), so
# each station contributes one number per model before aggregating across
# stations — this keeps seed variance and station-to-station variance separate.
per_station = (
    results_df.groupby(["Target", "Model", "HeldOutStation"])[["MAE", "RMSE", "R2"]]
    .mean()
    .reset_index()
)
per_station.to_csv(OUT_DIR / "experiment3_per_station_seedavg.csv", index=False)

summary_rows = []
for (target_name, model_name), grp in per_station.groupby(["Target", "Model"]):
    summary_rows.append({
        "Target": target_name,
        "Model": model_name,
        "MAE_mean": grp["MAE"].mean(), "MAE_sd": grp["MAE"].std(),
        "MAE_median": grp["MAE"].median(),
        "MAE_q1": grp["MAE"].quantile(0.25), "MAE_q3": grp["MAE"].quantile(0.75),
        "RMSE_mean": grp["RMSE"].mean(), "RMSE_sd": grp["RMSE"].std(),
        "RMSE_median": grp["RMSE"].median(),
        "RMSE_q1": grp["RMSE"].quantile(0.25), "RMSE_q3": grp["RMSE"].quantile(0.75),
        "R2_mean": grp["R2"].mean(), "R2_sd": grp["R2"].std(),
        "R2_median": grp["R2"].median(),
        "R2_q1": grp["R2"].quantile(0.25), "R2_q3": grp["R2"].quantile(0.75),
    })

summary_df = pd.DataFrame(summary_rows).round(4)
summary_df.to_csv(OUT_DIR / "experiment3_summary.csv", index=False)

print("\n=== Experiment 3 Summary (mean +/- SD across 50 stations, MAE/RMSE led) ===")
for _, row in summary_df.iterrows():
    print(f"[{row['Target']}] {row['Model']:16s} | "
          f"MAE={row['MAE_mean']:.3f}±{row['MAE_sd']:.3f}  "
          f"RMSE={row['RMSE_mean']:.3f}±{row['RMSE_sd']:.3f}  "
          f"R2={row['R2_mean']:.3f}±{row['R2_sd']:.3f} "
          f"(median R2={row['R2_median']:.3f}, IQR=[{row['R2_q1']:.3f}, {row['R2_q3']:.3f}])")

print(f"\nSaved summary to {OUT_DIR / 'experiment3_summary.csv'}")
print(f"Saved per-station (seed-averaged) results to {OUT_DIR / 'experiment3_per_station_seedavg.csv'}")
print("\nNOTE: These results are NOT directly comparable to Experiment 1 — "
      "Experiment 1 trains on all 50 stations; this experiment trains on 49 "
      "stations per fold. They answer different generalization questions "
      "(temporal vs. spatial) and must be reported/discussed separately.")

# ---- Figure: MAE/RMSE boxplot across 50 stations, per model, per target ----
# MAE/RMSE lead the reporting (per protocol) since per-station R2 can be
# unstable/long-tailed when a held-out station has low test-window variance.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

for col, target_name in enumerate(TARGETS.keys()):
    sub = per_station[per_station["Target"] == target_name]
    for row, metric in enumerate(["MAE", "RMSE"]):
        ax = axes[row, col]
        data = [sub[sub["Model"] == m][metric].values for m in MODEL_NAMES]
        ax.boxplot(data, tick_labels=MODEL_NAMES)
        ax.set_title(f"{target_name} — {metric} across 50 held-out stations")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
fig_path = OUT_DIR / "experiment3_loso_boxplot.png"
plt.savefig(fig_path, dpi=150)
print(f"Saved figure to {fig_path}")

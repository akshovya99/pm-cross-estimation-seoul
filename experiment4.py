"""
Experiment 4 — Stochasticity / Randomness (standalone, on the STANDARD split)

This was previously conflated with the seed variation embedded inside
Experiment 3's LOSO folds -- but those seeds vary jointly with the spatial
fold (49 training stations, held-out station), which is NOT the same
question as "how much does randomness alone affect results on the standard,
fixed, all-50-stations train/test split from Experiment 1?" This script
answers that separate, originally-scoped question directly.

Design:
  - Same train/test split as Experiment 1 (Jan 8-Oct 14 train / Oct 15-Dec 31
    test, all 50 stations in both).
  - Same reduced feature set, same fixed hyperparameters (no tuning).
  - RandomForest, LightGBM, XGBoost run across N_SEEDS different random
    seeds each; Linear Regression is deterministic and run once (no seed
    variance to report).
  - Reports mean +/- SD AND full per-seed distributions for MAE/RMSE/R2.
  - Per protocol: overlapping standard deviations do NOT by themselves prove
    model equivalence -- this script reports the distributions plainly and
    leaves the interpretive claim to the Discussion, rather than asserting
    statistical equivalence from SD overlap alone.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(".")
OUT_DIR = Path("./experiment4_out")
OUT_DIR.mkdir(exist_ok=True)

N_SEEDS = 10  # documented acceptable range: 5-10; using the upper end since
              # this is a single standard-split run, far cheaper than LOSO.
SEEDS = list(range(N_SEEDS))

train = pd.read_csv(DATA_DIR / "reduced_2024_train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "reduced_2024_test.csv", parse_dates=["Date"])

TARGETS = {
    "PM2.5": ["PM10", "WindSpeed", "Humidity"],
    "PM10": ["PM2.5", "WindSpeed", "Humidity"],
}

STOCHASTIC_MODELS = ["RandomForest", "LightGBM", "XGBoost"]
DETERMINISTIC_MODELS = ["LinearRegression"]

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

results = []

for target_name, feature_cols in TARGETS.items():
    X_train_raw = train[feature_cols].values
    y_train = train[target_name].values
    X_test_raw = test[feature_cols].values
    y_test = test[target_name].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Deterministic model: run once, no seed variance
    for model_name in DETERMINISTIC_MODELS:
        model = make_model(model_name, seed=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            "Target": target_name, "Model": model_name, "Seed": "N/A (deterministic)",
            "MAE": mae, "RMSE": rmse, "R2": r2,
        })
        print(f"[{target_name}] {model_name:16s} (deterministic) | MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}")

    # Stochastic models: run across N_SEEDS seeds
    for model_name in STOCHASTIC_MODELS:
        for seed in SEEDS:
            model = make_model(model_name, seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({
                "Target": target_name, "Model": model_name, "Seed": seed,
                "MAE": mae, "RMSE": rmse, "R2": r2,
            })
        # Print summary for this model after all seeds done
        model_rows = [r for r in results if r["Target"] == target_name and r["Model"] == model_name]
        maes = [r["MAE"] for r in model_rows]
        r2s = [r["R2"] for r in model_rows]
        print(f"[{target_name}] {model_name:16s} ({N_SEEDS} seeds) | "
              f"MAE={np.mean(maes):.3f}±{np.std(maes):.3f}  R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "experiment4_stochasticity_raw.csv", index=False)

# ---- Summary: mean +/- SD per target/model ----
summary_rows = []
for (target_name, model_name), grp in results_df.groupby(["Target", "Model"]):
    summary_rows.append({
        "Target": target_name,
        "Model": model_name,
        "N_seeds": len(grp) if model_name in STOCHASTIC_MODELS else 1,
        "MAE_mean": grp["MAE"].mean(), "MAE_sd": grp["MAE"].std() if len(grp) > 1 else 0.0,
        "RMSE_mean": grp["RMSE"].mean(), "RMSE_sd": grp["RMSE"].std() if len(grp) > 1 else 0.0,
        "R2_mean": grp["R2"].mean(), "R2_sd": grp["R2"].std() if len(grp) > 1 else 0.0,
    })
summary_df = pd.DataFrame(summary_rows).round(5)
summary_df.to_csv(OUT_DIR / "experiment4_summary.csv", index=False)

print("\n=== Experiment 4 Summary ===")
print(summary_df.to_string(index=False))
print(f"\nSaved raw per-seed results to {OUT_DIR / 'experiment4_stochasticity_raw.csv'}")
print(f"Saved summary to {OUT_DIR / 'experiment4_summary.csv'}")

# ---- Figure: R2 distribution across seeds, per model, per target ----
# This is also the source of error bars referenced for Experiment 1's figure.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, target_name in zip(axes, TARGETS.keys()):
    sub = results_df[(results_df["Target"] == target_name) & (results_df["Model"].isin(STOCHASTIC_MODELS))]
    data = [sub[sub["Model"] == m]["R2"].values for m in STOCHASTIC_MODELS]
    ax.boxplot(data, tick_labels=STOCHASTIC_MODELS)
    # Overlay deterministic LR as a horizontal reference line
    lr_r2 = results_df[(results_df["Target"] == target_name) & (results_df["Model"] == "LinearRegression")]["R2"].values[0]
    ax.axhline(lr_r2, color="red", linestyle="--", label=f"LinearRegression (deterministic, R²={lr_r2:.3f})")
    ax.set_title(f"{target_name} — R² across {N_SEEDS} seeds")
    ax.set_ylabel("R²")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path = OUT_DIR / "experiment4_stochasticity_figure.png"
plt.savefig(fig_path, dpi=150)
print(f"Saved figure to {fig_path}")

print("\nNOTE per protocol: overlapping standard deviations across models do "
      "NOT by themselves prove model equivalence. Report distributions and "
      "describe practical/effect-size differences in the Discussion rather "
      "than asserting statistical equivalence from SD overlap alone.")

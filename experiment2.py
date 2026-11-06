"""
Experiment 2 — Feature Ablation
Tests whether all three features are needed, by evaluating each model on
progressively larger feature subsets. Directly validates the "minimal-sensor"
claim with evidence rather than assumption.

Ablation steps (mirrored for both targets):
  PM2.5 target: A) PM10 only -> B) PM10+Wind -> C) PM10+Humidity -> D) PM10+Wind+Humidity
  PM10  target: A) PM2.5 only -> B) PM2.5+Wind -> C) PM2.5+Humidity -> D) PM2.5+Wind+Humidity

Protocol (per project plan §6):
  - Min-Max normalization fit on TRAIN ONLY, applied to test, refit per
    feature subset (since the feature set changes at every ablation step)
  - Same fixed hyperparameters as Experiment 1, no tuning
  - Same train/test split as Experiment 1 (Jan 8-Oct 14 train / Oct 15-Dec 31 test)
"""

import pandas as pd
import numpy as np
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
OUT_DIR = Path("./experiment2_out")
OUT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "reduced_2024_train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "reduced_2024_test.csv", parse_dates=["Date"])

# ---- Ablation steps per target ----
ABLATION = {
    "PM2.5": {
        "A_PM10_only": ["PM10"],
        "B_PM10_Wind": ["PM10", "WindSpeed"],
        "C_PM10_Humidity": ["PM10", "Humidity"],
        "D_PM10_Wind_Humidity": ["PM10", "WindSpeed", "Humidity"],
    },
    "PM10": {
        "A_PM25_only": ["PM2.5"],
        "B_PM25_Wind": ["PM2.5", "WindSpeed"],
        "C_PM25_Humidity": ["PM2.5", "Humidity"],
        "D_PM25_Wind_Humidity": ["PM2.5", "WindSpeed", "Humidity"],
    },
}

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

for target_name, steps in ABLATION.items():
    y_train = train[target_name].values
    y_test = test[target_name].values

    for step_name, feature_cols in steps.items():
        X_train_raw = train[feature_cols].values
        X_test_raw = test[feature_cols].values

        # Refit scaler for this specific feature subset, train only
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        models = make_models()

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Target": target_name,
                "AblationStep": step_name,
                "NumFeatures": len(feature_cols),
                "Features": ", ".join(feature_cols),
                "Model": model_name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            })

            print(f"[{target_name}] {step_name:24s} {model_name:16s} | "
                  f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "experiment2_ablation_results.csv", index=False)

print("\n=== Experiment 2 Ablation Results ===")
print(results_df.to_string(index=False))
print(f"\nSaved results to {OUT_DIR / 'experiment2_ablation_results.csv'}")

# ---- Figure: grouped bar chart, R2 per ablation step per model ----
# A bar chart (not a line plot) avoids ambiguity when two different feature
# combinations share the same feature COUNT (e.g. Step B = +Wind vs Step C =
# +Humidity both have 2 features) — each step gets its own labeled x-position.
STEP_LABELS = {
    "A_PM10_only": "PM10\nonly",
    "B_PM10_Wind": "+ Wind",
    "C_PM10_Humidity": "+ Humidity",
    "D_PM10_Wind_Humidity": "+ Wind\n+ Humidity",
    "A_PM25_only": "PM2.5\nonly",
    "B_PM25_Wind": "+ Wind",
    "C_PM25_Humidity": "+ Humidity",
    "D_PM25_Wind_Humidity": "+ Wind\n+ Humidity",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, target_name in zip(axes, ABLATION.keys()):
    sub = results_df[results_df["Target"] == target_name]
    step_order = list(ABLATION[target_name].keys())  # A, B, C, D in defined order
    models = sub["Model"].unique()
    n_models = len(models)
    n_steps = len(step_order)
    bar_width = 0.8 / n_models
    x = np.arange(n_steps)

    for i, model_name in enumerate(models):
        model_sub = sub[sub["Model"] == model_name].set_index("AblationStep").loc[step_order]
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, model_sub["R2"], width=bar_width, label=model_name)

    ax.set_title(f"Feature Ablation — {target_name}")
    ax.set_xlabel("Feature combination")
    ax.set_ylabel("R²")
    ax.set_xticks(x)
    ax.set_xticklabels([STEP_LABELS[s] for s in step_order])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

plt.tight_layout()
fig_path = OUT_DIR / "experiment2_ablation_figure.png"
plt.savefig(fig_path, dpi=150)
print(f"Saved figure to {fig_path}")

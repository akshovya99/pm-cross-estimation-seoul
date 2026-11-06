"""
Experiment 6 — Interpretability (SHAP dependence plot)
[Colab version — run on Google Colab due to a local Windows Smart App
Control policy blocking numba, a SHAP dependency, on the author's machine.
Results are unaffected by this: SHAP values depend only on the trained
model and data, not the compute platform, as confirmed by Experiment 1
producing identical accuracy metrics across sandbox, Colab, and local runs.]

Decision (per project plan, made after reviewing Experiments 1-3 results):
LightGBM was consistently one of the stronger, more stable models across
Experiment 1 (temporal) and Experiment 3 (spatial LOSO), and showed
non-obvious behavior in the Experiment 2 ablation. It is the model
interrogated here.

Per protocol: a SHAP DEPENDENCE plot is used as the primary figure, not a
SHAP summary plot — on a 3-feature model, a summary plot is largely redundant
with a standard feature-importance ranking. A dependence plot instead shows
the SHAPE of each feature's relationship with the prediction (e.g. linear,
threshold, saturating) and reveals interaction effects (via color-coding by
a second feature) — genuinely new information beyond a ranking.

This trains fresh on the full reduced feature set (same as Experiment 1) for
both targets, then computes SHAP values on the test set.

To run: requires reduced_2024_train.csv and reduced_2024_test.csv in the
same folder as DATA_DIR points to (adjust DATA_DIR/OUT_DIR if running
locally instead of on Colab/Drive).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("/content/drive/My Drive/experiments")
OUT_DIR = Path("/content/drive/My Drive/experiments/experiment6_out")
OUT_DIR.mkdir(exist_ok=True)

train = pd.read_csv(DATA_DIR / "reduced_2024_train.csv", parse_dates=["Date"])
test = pd.read_csv(DATA_DIR / "reduced_2024_test.csv", parse_dates=["Date"])

TARGETS = {
    "PM2.5": ["PM10", "WindSpeed", "Humidity"],
    "PM10": ["PM2.5", "WindSpeed", "Humidity"],
}

for target_name, feature_cols in TARGETS.items():
    X_train_raw = train[feature_cols]
    y_train = train[target_name].values
    X_test_raw = test[feature_cols]
    y_test = test[target_name].values

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_cols)

    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # ---- SHAP values on the test set ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save raw SHAP values for reproducibility / re-plotting
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{c}" for c in feature_cols])
    shap_df = pd.concat([X_test_raw.reset_index(drop=True), shap_df], axis=1)
    shap_df.to_csv(OUT_DIR / f"experiment6_shap_values_{target_name.replace('.', '')}.csv", index=False)

    # ---- Primary figure: dependence plot ----
    # Main feature = the dominant predictor (PM10 for PM2.5 target, PM2.5 for PM10 target)
    # Colored by wind speed to reveal potential interaction effects.
    main_feature = feature_cols[0]  # PM10 or PM2.5 — the dominant predictor
    color_feature = "WindSpeed"

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        main_feature,
        shap_values,
        X_test_raw,  # use raw (unscaled) values on x-axis for interpretability
        interaction_index=color_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence — {main_feature} vs. predicted {target_name}\n(colored by {color_feature})")
    plt.tight_layout()
    fig_path = OUT_DIR / f"experiment6_shap_dependence_{target_name.replace('.', '')}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[{target_name}] Saved SHAP dependence plot to {fig_path}")

    # ---- Secondary, smaller figure: mean |SHAP value| bar (feature ranking) ----
    # Kept small/secondary since Experiment 2's ablation already established
    # feature relevance — this is just a sanity-check cross-reference, not a
    # headline figure.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.barh(feature_cols, mean_abs_shap)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title(f"{target_name} — feature relevance (LightGBM, SHAP)")
    plt.tight_layout()
    fig2_path = OUT_DIR / f"experiment6_shap_ranking_{target_name.replace('.', '')}.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"[{target_name}] Saved SHAP ranking (secondary) plot to {fig2_path}")

print("\nExperiment 6 complete.")
print("Primary figures (use in paper): experiment6_shap_dependence_*.png")
print("Secondary/reference figures: experiment6_shap_ranking_*.png")

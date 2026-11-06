# PM2.5/PM10 cross-estimation in Seoul (2024)

Code and data for the manuscript "Contemporaneous PM Cross-Estimation in Seoul: Minimal Features, Models & Deployment Trade-offs" (Shrestha, Nguyen & Bhandari).

## Setup

Tested with Python 3.14.7 (Python 3.9+ should work). Package versions are listed in `requirements.txt`.

    pip install -r requirements.txt

## How to run

Run each script from this folder, for example:

    python experiment1.py

## Scripts

| Script | Evaluation (paper objective) | Output folder |
|---|---|---|
| experiment1.py | Standard temporal evaluation (O1) | experiment1_out |
| experiment2.py | Feature ablation (O2) | experiment2_out |
| experiment4.py | Stochasticity, 10 seeds (O3) | experiment4_out |
| experiment3.py | Leave-one-station-out validation (O4) | experiment3_out |
| experiment5_deployment.py | Raspberry Pi 4 benchmark (O5) | experiment5_out |
| experiment6_colab.py | SHAP interpretability (O6) | experiment6_out |

## Protocol (all experiments)

- Reduced 3-feature set: cross-pollutant (PM10 for PM2.5, PM2.5 for PM10), wind speed and humidity.
- Train/test split: train 8 Jan-14 Oct 2024 (14,050 station-days), test 15 Oct-31 Dec 2024 (3,900 station-days).
- Min-Max normalisation fitted on the training data only and applied to the test data.
- Fixed hyperparameters, no test-set tuning:
  - Random Forest: 100 trees, max depth 8, min_samples_leaf 3 (as in the conference paper)
  - LightGBM and XGBoost: library defaults
  - Linear Regression: defaults

## Experiment 1 details

Trains Linear Regression, Random Forest, LightGBM and XGBoost for PM2.5 and PM10 and reports MAE, RMSE and R2, batch predict() time on the full test set, and serialised model size (KB).

Outputs: `experiment1_out/experiment1_results.csv` and 8 trained models (`*.joblib`, one per target and model), which the Raspberry Pi benchmark (`experiment5_deployment.py`) reuses.

Expected results for sanity checking: Random Forest is strongest for PM2.5 (R2 about 0.80) and Linear Regression is best for PM10 (R2 about 0.80). Exact values can differ slightly between library versions.

## Notes

- `experiment6_colab.py` was run in Google Colab with the CSV files in `My Drive/experiments`. To run it locally, set `DATA_DIR = Path(".")` and `OUT_DIR = Path("experiment6_out")`.
- `experiment5_deployment.py` was run on a Raspberry Pi 4 (4 GB RAM, Debian 13, aarch64). The exact environment is saved in `experiment5_out/experiment5_environment.json`.

## Data files

- `reduced_2024_full.csv`: all 17,950 station-day records (50 stations, 8 Jan-31 Dec 2024)
- `reduced_2024_train.csv` and `reduced_2024_test.csv`: the two partitions

Sources: PM10 and PM2.5 from the Seoul Open Data Plaza (https://data.seoul.go.kr/); wind speed and humidity from Open-Meteo (https://open-meteo.com/), one central-Seoul coordinate, identical for all stations on a given date.

## License

MIT (see `LICENSE`).

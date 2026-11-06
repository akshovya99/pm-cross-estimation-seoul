"""
Experiment 5 — Raspberry Pi 4 Deployment Benchmarking (native Python)

Loads the 8 models trained in Experiment 1 and re-runs inference on the Pi,
using the SAME test set, to measure deployment cost (speed + memory) on
constrained hardware. Accuracy (MAE/RMSE/R2) is expected to be identical or
near-identical to Experiment 1's laptop numbers, since these are the exact
same trained model weights — the comparison here is about SPEED and MEMORY,
not accuracy.

Environment (recorded per project plan §6, confirmed before running this):
  Device:        Raspberry Pi 4 Model B, 4 GB RAM
  OS:            Debian GNU/Linux 13 (trixie), kernel 6.18.29+rpt-rpi-v8
  Architecture:  aarch64 (64-bit)
  CPU governor:  performance (all 4 cores, set immediately before this run)
  Clock speed:   1.8 GHz
  Idle temp:     39.4 degC (measured just before running)
  Python:        3.13.5

Protocol:
  - Batch predict() on the FULL test set, timed with time.perf_counter_ns(),
    same timing style as Experiment 1 (never mixed with per-sample timing).
    One untimed WARM-UP call precedes the timed repeats, since the first
    call can carry one-time initialization overhead (especially noticeable
    for LightGBM/XGBoost) unrelated to steady-state inference speed.
  - Memory reported as RSS DELTA (process memory with model loaded minus
    baseline interpreter memory), not raw RSS -- raw RSS is dominated by the
    Python interpreter itself and misrepresents actual model deployment cost.
    The baseline is re-established (with an explicit gc.collect()) fresh
    before EACH model is loaded, and each model is freed afterward, so every
    model's RSS delta reflects only its own footprint, not cumulative growth
    from previously loaded models in the same run.
  - No ONNX Runtime in this scope (kept in Future Work) -- native Python only.

No re-training happens here -- models are loaded directly from the .joblib
files produced in Experiment 1, so accuracy metrics use the exact same
learned weights as the laptop run.
"""

"""
Experiment 5 — Raspberry Pi 4 Deployment Benchmarking (native Python)

Loads the 8 models trained in Experiment 1 and re-runs inference on the Pi,
using the SAME test set, to measure deployment cost (speed + memory) on
constrained hardware. Accuracy (MAE/RMSE/R2) is expected to be identical or
near-identical to Experiment 1's laptop numbers, since these are the exact
same trained model weights -- the comparison here is about SPEED and MEMORY,
not accuracy.

Environment (recorded per project plan §6, confirmed before running this):
  Device:        Raspberry Pi 4 Model B, 4 GB RAM
  OS:            Debian GNU/Linux 13 (trixie), kernel 6.18.29+rpt-rpi-v8
  Architecture:  aarch64 (64-bit)
  CPU governor:  performance (all 4 cores, set immediately before this run)
  Clock speed:   1.8 GHz
  Idle temp:     39.4 degC (measured just before running)
  Python:        3.13.5

Protocol:
  - Batch predict() on the FULL test set, timed with time.perf_counter_ns(),
    same timing style as Experiment 1 (never mixed with per-sample timing).
    One untimed WARM-UP call precedes the timed repeats, since the first
    call can carry one-time initialization overhead (especially noticeable
    for LightGBM/XGBoost) unrelated to steady-state inference speed.
  - Memory reported as RSS DELTA (process memory with model loaded minus
    baseline interpreter memory), not raw RSS -- raw RSS is dominated by the
    Python interpreter itself and misrepresents actual model deployment cost.

    IMPORTANT (fixed after a prior run showed implausible near-zero deltas
    for the second target's models): measuring multiple models sequentially
    within ONE process, even with gc.collect() between them, is unreliable
    on Linux/glibc, because glibc's malloc typically retains freed heap
    pages in the process's arena for reuse rather than returning them to
    the OS -- so RSS does not actually shrink after a model is deleted, and
    a later, smaller model can silently reuse already-resident freed memory,
    producing an artificially small apparent delta. To avoid this, EACH
    model now runs in its OWN FRESH SUBPROCESS (via multiprocessing), so
    every measurement starts from a genuinely cold process baseline.
  - No ONNX Runtime in this scope (kept in Future Work) -- native Python only.

No re-training happens here -- models are loaded directly from the .joblib
files produced in Experiment 1, so accuracy metrics use the exact same
learned weights as the laptop run.
"""

import time
import json
import joblib
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path(".")
MODEL_DIR = Path("./experiment1_out")
OUT_DIR = Path("./experiment5_out")
OUT_DIR.mkdir(exist_ok=True)

TARGETS = {
    "PM2.5": ["PM10", "WindSpeed", "Humidity"],
    "PM10": ["PM2.5", "WindSpeed", "Humidity"],
}
MODEL_NAMES = ["LinearRegression", "RandomForest", "LightGBM", "XGBoost"]
N_REPEATS = 20


def _run_single_model(target_name, feature_cols, model_name, data_dir, model_dir, result_queue):
    """
    Runs in its OWN fresh subprocess. Measures this process's own baseline
    RSS before loading anything, then the delta after loading the model --
    guaranteed cold, since this process has touched nothing else.
    """
    import joblib as _joblib
    import psutil as _psutil
    import numpy as _np
    import pandas as _pd
    import time as _time
    from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
    from sklearn.metrics import mean_absolute_error as _mae, mean_squared_error as _mse, r2_score as _r2

    proc = _psutil.Process()
    baseline_rss_mb = proc.memory_info().rss / (1024 * 1024)

    train = _pd.read_csv(data_dir / "reduced_2024_train.csv", parse_dates=["Date"])
    test = _pd.read_csv(data_dir / "reduced_2024_test.csv", parse_dates=["Date"])

    X_train_raw = train[feature_cols].values
    X_test_raw = test[feature_cols].values
    y_test = test[target_name].values

    scaler = _MinMaxScaler()
    scaler.fit(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model_path = model_dir / f"{target_name}_{model_name}.joblib"
    model = _joblib.load(model_path)

    rss_after_load_mb = proc.memory_info().rss / (1024 * 1024)
    rss_delta_mb = rss_after_load_mb - baseline_rss_mb

    # Warm-up call (untimed) then timed repeats
    _ = model.predict(X_test)
    times_ms = []
    y_pred = None
    for _i in range(N_REPEATS):
        t0 = _time.perf_counter_ns()
        y_pred = model.predict(X_test)
        t1 = _time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)

    mae = _mae(y_test, y_pred)
    rmse = _np.sqrt(_mse(y_test, y_pred))
    r2 = _r2(y_test, y_pred)

    result_queue.put({
        "Target": target_name,
        "Model": model_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Predict_Time_mean_ms": round(float(_np.mean(times_ms)), 4),
        "Predict_Time_sd_ms": round(float(_np.std(times_ms)), 4),
        "Baseline_RSS_MB": round(baseline_rss_mb, 3),
        "RSS_after_load_MB": round(rss_after_load_mb, 3),
        "RSS_delta_MB": round(rss_delta_mb, 3),
    })


if __name__ == "__main__":
    results = []
    for target_name, feature_cols in TARGETS.items():
        for model_name in MODEL_NAMES:
            model_path = MODEL_DIR / f"{target_name}_{model_name}.joblib"
            if not model_path.exists():
                print(f"WARNING: {model_path} not found, skipping.")
                continue

            result_queue = mp.Queue()
            p = mp.Process(
                target=_run_single_model,
                args=(target_name, feature_cols, model_name, DATA_DIR, MODEL_DIR, result_queue),
            )
            p.start()
            row = result_queue.get()
            p.join()

            results.append(row)
            print(f"[{row['Target']}] {row['Model']:16s} | MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} "
                  f"R2={row['R2']:.4f} | predict={row['Predict_Time_mean_ms']:.3f}"
                  f"±{row['Predict_Time_sd_ms']:.3f}ms | "
                  f"baseline={row['Baseline_RSS_MB']:.2f}MB -> "
                  f"loaded={row['RSS_after_load_MB']:.2f}MB "
                  f"(delta={row['RSS_delta_MB']:.2f}MB)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "experiment5_pi_results.csv", index=False)

    env_info = {
        "device": "Raspberry Pi 4 Model B, 4 GB RAM",
        "os": "Debian GNU/Linux 13 (trixie), kernel 6.18.29+rpt-rpi-v8",
        "architecture": "aarch64",
        "cpu_governor": "performance",
        "clock_speed_ghz": 1.8,
        "idle_temp_c": 39.4,
        "python_version": "3.13.5",
        "n_repeats_per_prediction": N_REPEATS,
        "rss_measurement_method": "one fresh subprocess per model (avoids glibc "
                                   "malloc-arena retention artifacts from sequential "
                                   "in-process measurement)",
    }
    with open(OUT_DIR / "experiment5_environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    print(f"\nSaved results to {OUT_DIR / 'experiment5_pi_results.csv'}")
    print(f"Saved environment metadata to {OUT_DIR / 'experiment5_environment.json'}")
    print("\n=== Experiment 5 Results (Raspberry Pi 4, native Python, subprocess-isolated) ===")
    print(results_df.to_string(index=False))

"""Generate the synthetic battery dataset and train XGBoost + Random Forest.

Run manually with:  python train.py
(engine.py also calls this automatically the first time, if models/ is empty.)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURES = ["cycle", "soc", "voltage", "current", "temperature", "capacity", "fast_charging"]
TARGET = "soh"


def generate_dataset(n: int = 7000, seed: int = 42) -> pd.DataFrame:
    """Physics-inspired battery ageing simulation (NASA PCoE-style behaviour)."""
    rng = np.random.default_rng(seed)

    cycle = rng.uniform(0, 600, n)
    soc = rng.uniform(10, 100, n)
    current = rng.uniform(0.5, 3.0, n)
    temperature = rng.uniform(10.0, 55.0, n)
    fast = rng.binomial(1, 0.4, n)

    # Terminal voltage = open-circuit voltage - I*R (resistance grows with age / cold)
    r_int = 0.08 + (cycle / 600.0) * 0.06 + np.where(temperature < 18, (18 - temperature) * 0.003, 0.0)
    voltage = np.clip(15.0 + (soc / 100.0) * 3.3 - current * r_int + rng.normal(0, 0.08, n), 14.0, 18.5)

    # Capacity fade (percentage points of SoH lost)
    hot = np.maximum(0, temperature - 25.0)
    cold = np.maximum(0, 15.0 - temperature)
    degradation = (
        0.055 * cycle + 0.000035 * cycle**2          # cycle ageing
        + fast * 0.025 * cycle                        # DC fast-charge stress
        + 0.28 * hot + 0.15 * cold                    # thermal stress
        + (current > 2.0) * (hot > 10.0) * 2.5        # high current + heat
        + np.maximum(0, 15.2 - voltage) * 1.8         # deep discharge
        + rng.normal(0, 0.6, n)
    )
    soh = np.clip(100.0 - degradation, 45.0, 99.8)

    fresh = rng.normal(10.0, 0.25, n)
    capacity = np.clip(soh / 100.0 * fresh + rng.normal(0, 0.05, n), 5.5, 12.0)

    return pd.DataFrame({
        "cycle": cycle.round(1), "soc": soc.round(1), "voltage": voltage.round(2),
        "current": current.round(2), "temperature": temperature.round(1),
        "capacity": capacity.round(2), "fast_charging": fast, "soh": soh.round(2),
    })


def _metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_save(models_dir: str = MODELS_DIR) -> dict:
    os.makedirs(models_dir, exist_ok=True)
    df = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    metrics = {}

    print("[train] Random Forest ...")
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_split=4,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    metrics["rf"] = _metrics(y_test, rf.predict(X_test))
    joblib.dump(rf, os.path.join(models_dir, "rf_model.joblib"), compress=3)

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[train] xgboost is not installed -> skipping XGBoost (pip install xgboost)")
    else:
        print("[train] XGBoost ...")
        xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.07,
                           subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        metrics["xgboost"] = _metrics(y_test, xgb.predict(X_test))
        joblib.dump(xgb, os.path.join(models_dir, "xgb_model.joblib"), compress=3)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    for name, m in metrics.items():
        print(f"[train] {name:8s} R2={m['r2']:.4f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}")
    return metrics


if __name__ == "__main__":
    train_and_save()

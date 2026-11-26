"""Model loading + inference.

FIELDS is the single source of truth for the input sliders: the HTML template
builds the sliders from it and the API validates against it, so the frontend
and backend can never disagree about ranges, steps or defaults.
"""
import json
import math
import os
import threading
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from train import FEATURES, MODELS_DIR, train_and_save

# ---------------------------------------------------------------- input fields
FIELDS = [
    dict(key="cycle", label="Charge Cycles", min=0, max=600, step=1, default=20,
         decimals=0, unit="", hint="0 – new battery | 600 – end of life"),
    dict(key="soc", label="State of Charge (%)", min=10, max=100, step=1, default=80,
         decimals=0, unit=" %", hint=""),
    dict(key="voltage", label="Voltage (V)", min=14.0, max=18.5, step=0.1, default=17.5,
         decimals=1, unit=" V", hint=""),
    dict(key="current", label="Discharge Current (A)", min=0.5, max=3.0, step=0.1, default=1.0,
         decimals=1, unit=" A", hint="Higher current → faster degradation"),
    dict(key="temperature", label="Temperature (°C)", min=10, max=55, step=1, default=20,
         decimals=0, unit=" °C", hint="Optimal 15–30°C | >40°C = thermal stress"),
    dict(key="capacity", label="Battery Capacity (Ah)", min=5.5, max=12.0, step=0.1, default=9.5,
         decimals=1, unit=" Ah", hint="Fresh battery: ~10 Ah"),
]
DEFAULTS = {f["key"]: f["default"] for f in FIELDS}

MODEL_FILES = {"xgboost": "xgb_model.joblib", "rf": "rf_model.joblib"}
DEFAULT_MODEL = "xgboost"
CURVE_CYCLES = (0, 100, 200, 300, 400, 500, 600)

EOL_SOH = 80.0            # end-of-life threshold (%)
PACK_BASE_VALUE = 5333    # pack value at 100 % SoH (USD); 40 kWh x $120/kWh benchmark

# -------------------------------------------------------------------- loading
_lock = threading.Lock()
models: dict = {}


def _load_all() -> None:
    def load(name):
        model = joblib.load(os.path.join(MODELS_DIR, MODEL_FILES[name]))
        model.set_params(n_jobs=1)        # tiny batches: threads only add overhead
        return model

    missing = [f for f in MODEL_FILES.values() if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print("[engine] No trained models found - training now (one-time, ~10 s) ...")
        train_and_save()

    for name in MODEL_FILES:
        try:
            models[name] = load(name)
        except FileNotFoundError:
            pass                                       # e.g. xgboost not installed
        except Exception as exc:                       # stale pickle / version mismatch
            print(f"[engine] Could not load {name} ({exc!r}) - retraining ...")
            train_and_save()
            try:
                models[name] = load(name)
            except Exception as exc2:
                print(f"[engine] {name} unavailable: {exc2!r}")

    if not models:
        raise RuntimeError("No model could be loaded. Run: pip install -r requirements.txt")
    print(f"[engine] Ready. Models: {', '.join(models)}")


# ----------------------------------------------------------------- validation
def parse_inputs(payload) -> dict:
    """Validate + clamp a request body. Raises ValueError on bad input."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    values = {}
    for f in FIELDS:
        raw = payload.get(f["key"], f["default"])
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{f['key']}' must be a number.") from None
        if not math.isfinite(v):
            raise ValueError(f"'{f['key']}' must be a finite number.")
        values[f["key"]] = round(min(max(v, f["min"]), f["max"]), 3)
    values["fast_charging"] = 1 if payload.get("fast_charging", False) in (True, 1, "true", "1") else 0
    return values


def _status(soh: float):
    if soh >= 90:
        return "Excellent", "green", "Battery is in excellent condition with high capacity."
    if soh >= 80:
        return "Good", "lime", "Battery is in good condition. Capacity is fading normally."
    if soh >= 70:
        return "Poor", "amber", "Battery is below the 80% end-of-life line. Plan a replacement."
    return "Critical", "red", "Battery is severely degraded. Replace it soon."


# ------------------------------------------------------------------ inference
@lru_cache(maxsize=8192)
def _infer(model_name, cycle, soc, voltage, current, temperature, capacity, fast):
    """One batched model call: row 0 = current state, rows 1..7 = degradation curve."""
    rows = [[cycle, soc, voltage, current, temperature, capacity, fast]]
    for c in CURVE_CYCLES:
        projected_capacity = max(6.0, capacity - (c - cycle) * 0.005)
        rows.append([c, soc, voltage, current, temperature, projected_capacity, fast])
    X = pd.DataFrame(rows, columns=FEATURES)

    with _lock:
        preds = np.clip(models[model_name].predict(X), 40.0, 99.9)

    soh = float(preds[0])
    curve = np.minimum.accumulate(preds[1:])          # batteries never get healthier

    degradation_rate = (0.065
                        * (1.0 + max(0.0, temperature - 25.0) * 0.02)
                        * (1.35 if fast else 1.0)
                        * (1.0 + (current - 1.0) * 0.1))
    rul = int(round((soh - EOL_SOH) / degradation_rate)) if soh > EOL_SOH else 0
    status, tone, message = _status(soh)

    return {
        "soh": round(soh, 2),
        "remaining_cycles": max(0, rul),
        "pack_value": int(round(soh / 100.0 * PACK_BASE_VALUE)),
        "curve": tuple(round(float(v), 2) for v in curve),
        "status": status,
        "tone": tone,
        "message": message,
        "model_used": model_name,
    }


def predict(payload: dict) -> dict:
    values = parse_inputs(payload)
    name = str(payload.get("model", DEFAULT_MODEL)).lower()
    if name not in models:
        name = DEFAULT_MODEL if DEFAULT_MODEL in models else next(iter(models))
    result = _infer(name, values["cycle"], values["soc"], values["voltage"], values["current"],
                    values["temperature"], values["capacity"], values["fast_charging"])
    return {**result, "curve": list(result["curve"])}


def metrics() -> dict:
    path = os.path.join(MODELS_DIR, "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_load_all()
predict({})        # warm-up: first real request is as fast as every other one

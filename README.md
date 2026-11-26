# ⚡ EV Battery Health Prediction

Predicts a battery's **State of Health (SoH)**, **Remaining Useful Life (RUL)** and **pack value**
from live telemetry, using **XGBoost** or **Random Forest**, with a live degradation curve.

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:8000**. The first start trains the models automatically (about 10 s,
saved in `models/`); every later start is instant.

## Project layout

```
app.py                 Flask server: page + REST API
engine.py              model loading, validation, inference, input ranges (single source of truth)
train.py               synthetic dataset + XGBoost / Random Forest training (python train.py)
templates/index.html   dashboard page
static/style.css, app.js   styling + live-sync logic (no external JS libraries)
tests/test_app.py      API tests:  python -m unittest discover -s tests -v
notebooks/             cross-validation notebook (for the report)
```

## Why the UI feels instant

- Slider ranges/steps/defaults come from `engine.FIELDS`, so the page and the API always agree.
- The page is rendered with the first prediction already inside it (no empty flash).
- One model call per request (current state + all 7 curve points in a single batch), models loaded
  once at startup, results cached on the server **and** in the browser.
- While you drag a slider, only one request is in flight; when it returns, the newest slider
  position is sent. The screen never lags behind and requests never pile up.
- The chart is hand-drawn SVG, so there is no Chart.js/Tailwind download.

## API

`POST /api/predict/`

```json
{ "cycle": 20, "soc": 80, "voltage": 17.5, "current": 1.0,
  "temperature": 20, "capacity": 9.5, "model": "xgboost", "fast_charging": false }
```
Returns `soh`, `remaining_cycles`, `pack_value`, `curve` (SoH at 0, 100 … 600 cycles), `status`
(Excellent / Good / Poor / Critical), `message`, `model_used`.
Out-of-range numbers are clamped; non-numeric input returns HTTP 400.
`GET /api/health/` lists loaded models and their test metrics.

## Notes

- The training data is **synthetic**, generated from a physics-inspired ageing model (cycle fade,
  heat/cold stress, fast charging, deep discharge) in the style of NASA PCoE behaviour. It is not
  the real NASA dataset. Say so in your report/viva.
- Test-set metrics are written to `models/metrics.json` after training (R² ≈ 0.995).
- Pack value = SoH × $5,333 (the original project's constant for a 40 kWh pack benchmark).

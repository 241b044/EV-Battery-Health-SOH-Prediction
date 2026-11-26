"""EV Battery Health Prediction - Flask app.

Run:  python app.py      then open http://127.0.0.1:8000
"""
import os

from flask import Flask, jsonify, render_template, request

import engine

app = Flask(__name__)
app.json.sort_keys = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 86400


def _asset_version() -> int:
    """Changes whenever a static file changes, so browsers never serve stale JS/CSS."""
    static = os.path.join(app.root_path, "static")
    return max(int(os.path.getmtime(os.path.join(static, f))) for f in os.listdir(static))


@app.get("/")
def index():
    initial = engine.predict({"model": engine.DEFAULT_MODEL})   # first paint already has data
    return render_template("index.html", fields=engine.FIELDS, r=initial,
                           models=list(engine.models), v=_asset_version())


@app.route("/api/predict/", methods=["POST"], strict_slashes=False)
def predict():
    payload = request.get_json(silent=True)
    try:
        return jsonify(engine.predict(payload))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/health/")
def health():
    return jsonify(status="healthy", models=list(engine.models), metrics=engine.metrics())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  EV Battery Health  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

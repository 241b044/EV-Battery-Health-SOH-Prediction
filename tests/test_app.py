import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine  # noqa: E402
from app import app  # noqa: E402

BASE = {"cycle": 20, "soc": 80, "voltage": 17.5, "current": 1.0,
        "temperature": 20, "capacity": 9.5}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, **extra):
        return self.client.post("/api/predict/", json={**BASE, **extra})

    def test_dashboard_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"EV Battery Health Prediction", res.data)
        self.assertIn(b"Predict Battery Health", res.data)

    def test_health(self):
        data = self.client.get("/api/health/").get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models"])

    def test_every_model_predicts(self):
        for name in engine.models:
            data = self.post(model=name).get_json()
            self.assertEqual(data["model_used"], name)
            self.assertEqual(len(data["curve"]), 7)
            self.assertTrue(40 <= data["soh"] <= 99.9)
            self.assertEqual(data["curve"], sorted(data["curve"], reverse=True))

    def test_no_trailing_slash(self):
        self.assertEqual(self.client.post("/api/predict", json=BASE).status_code, 200)

    def test_stress_lowers_health(self):
        good = self.post().get_json()["soh"]
        bad = self.post(cycle=450, temperature=48, current=2.8, voltage=14.8,
                        capacity=7.0, fast_charging=True).get_json()
        self.assertLess(bad["soh"], good)
        self.assertIn(bad["status"], ("Poor", "Critical"))
        self.assertEqual(bad["remaining_cycles"], 0)

    def test_out_of_range_is_clamped(self):
        self.assertEqual(self.post(cycle=99999).status_code, 200)

    def test_bad_input_rejected(self):
        self.assertEqual(self.post(cycle="abc").status_code, 400)
        self.assertEqual(self.client.post("/api/predict/", data="nope").status_code, 400)


if __name__ == "__main__":
    unittest.main()

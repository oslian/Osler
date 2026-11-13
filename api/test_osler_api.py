import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import adaptive_cell_weights as weights
from mimic_to_eicu_49_cell_external_audit import normalize_cohort
from osler_api import app as api
from osler_api import runtime
from osler_jepa.frozen_protocol import sha256
from test_adaptive_cell_weights import transitions

CELL = "potassium@6h"


def write_model(folder: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    """A weights folder and registry like the exported ones, with five synthetic potassium@6h bundles."""

    out = folder / "weights"
    out.mkdir()
    train = np.arange(len(frame))
    entries = []
    for sample in range(5):
        bundle = weights.fit_predictor(frame, train, target="potassium", recipe="measurement_pure_ridge",
                                       seed=11 + sample, policy=None)
        bundle = {**bundle, "cell": CELL, "sample": sample, "horizon_hours": 6}
        path = out / f"potassium_6h_s{sample}.joblib"
        joblib.dump(bundle, path)
        entries.append({"cell": CELL, "sample": sample, "file": path.name, "sha256": sha256(path)})
    registry = folder / "registry.json"
    registry.write_text(json.dumps({"cells": [
        {"cell": CELL, "target": "potassium", "horizon_hours": 6, "task_type": "continuous_value",
         "status": "independently_replicated", "recipe": "measurement_pure_ridge", "flags": ["not_replicated_at_X"]},
        {"cell": "map@6h", "target": "map", "horizon_hours": 6, "task_type": "continuous_value", "status": "candidate"},
    ]}), encoding="utf-8")
    import sklearn
    (out / "MANIFEST.json").write_text(json.dumps({
        "schema": "adaptive_cell_weights.v1", "registry_sha256": sha256(registry),
        "versions": {"scikit-learn": sklearn.__version__}, "bundles": entries}), encoding="utf-8")
    return out, registry


def anchors_of(raw: pd.DataFrame) -> list[dict]:
    values = raw.drop(columns=["subject_id", "stay_id", "t", "first_careunit", "potassium_tp6", "potassium_tp6_offset_hr"])
    return [{"time": time, "values": {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}}
            for time, (_, row) in zip(raw["t"], values.iterrows())]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = transitions(300, 0)
        frame, _ = normalize_cohort(cls.raw, database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        cls.frame = frame.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        cls.folder = Path(tempfile.mkdtemp())
        cls.weights, cls.registry = write_model(cls.folder, cls.frame)
        cls.model = runtime.load_model(cls.weights, cls.registry)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)

    def test_forecast_equals_the_frozen_scorer_path_at_every_anchor(self):
        stay = self.raw[self.raw["stay_id"] == 3].sort_values("t").reset_index(drop=True)
        scored = self.frame[self.frame["stay_id"] == "MIMIC:3"].set_index("anchor_time")
        checked = 0
        for k in range(len(stay)):
            if stay["t"][k] not in scored.index:
                continue
            result = runtime.forecast(self.model, anchors_of(stay.iloc[:k + 1]))["forecasts"][0]
            row = scored.loc[[stay["t"][k]]].reset_index()
            expected = [float(weights.predict(bundle, row)[0]) for bundle in self.model.cells[CELL].bundles]
            np.testing.assert_allclose(result["sample_points"], expected, rtol=0, atol=1e-9)
            self.assertEqual(result["point"], float(np.median(expected)))
            checked += 1
        self.assertGreater(checked, 3)

    def test_anchor_order_does_not_matter(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 4])
        self.assertEqual(runtime.forecast(self.model, anchors), runtime.forecast(self.model, anchors[::-1]))

    def test_caller_supplied_histories_are_recomputed(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 5])
        clean = runtime.forecast(self.model, anchors)
        for anchor in anchors:
            anchor["values"].update({"lag1__potassium_t": 99.0, "slope1h__potassium_t": 99.0, "hours_since_onset": 999.0})
        forged = runtime.forecast(self.model, anchors)
        self.assertEqual(clean["forecasts"], forged["forecasts"])
        self.assertEqual(forged["recomputed_columns"], ["hours_since_onset", "lag1__potassium_t", "slope1h__potassium_t"])

    def test_spacing_other_than_six_hours_is_rejected(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 6])
        anchors[1]["time"] = anchors[1]["time"] + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "6 hours apart"):
            runtime.forecast(self.model, anchors)
        with self.assertRaisesRegex(ValueError, "unique"):
            runtime.forecast(self.model, [anchors[0], anchors[0]])

    def test_missing_current_value_gives_no_point(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 7])
        anchors[-1]["values"]["potassium_t"] = None
        result = runtime.forecast(self.model, anchors)["forecasts"][0]
        self.assertEqual((result["status"], result["point"], result["interval"]), ("current_value_missing", None, None))

    def test_unknown_inputs_and_cells_are_reported(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 8])
        anchors[-1]["values"]["potasium_t"] = 4.0
        ignored = runtime.forecast(self.model, anchors)["ignored_columns"]
        self.assertIn("potasium_t", ignored)
        self.assertNotIn("potassium_t", ignored)
        with self.assertRaisesRegex(ValueError, "unknown cells"):
            runtime.forecast(self.model, anchors, ["map@6h"])

    def test_only_replicated_cells_are_served_with_their_flags(self):
        self.assertEqual(list(self.model.cells), [CELL])
        self.assertEqual(self.model.cells[CELL].flags, ("not_replicated_at_X",))
        self.assertEqual([b["sample"] for b in self.model.cells[CELL].bundles], [0, 1, 2, 3, 4])

    def test_loading_fails_closed_on_a_changed_bundle_or_registry(self):
        folder = Path(tempfile.mkdtemp())
        try:
            out, registry = write_model(folder, self.frame)
            with (out / "potassium_6h_s2.joblib").open("ab") as handle:
                handle.write(b"0")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                runtime.load_model(out, registry)
            registry.write_text(registry.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "registry"):
                runtime.load_model(out, registry)
        finally:
            shutil.rmtree(folder)


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = transitions(300, 1)
        frame, _ = normalize_cohort(cls.raw, database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        cls.folder = Path(tempfile.mkdtemp())
        cls.model = runtime.load_model(*write_model(cls.folder, frame.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder)

    def client(self):
        client = TestClient(api.app)                       # not entered, so the lifespan does not load the real weights
        api.app.state.expert = self.model
        return client

    def body(self):
        anchors = anchors_of(self.raw[self.raw["stay_id"] == 2])
        return {"anchors": [{"time": a["time"].isoformat(), "values": a["values"]} for a in anchors]}

    def test_forecast_endpoint_matches_the_runtime(self):
        response = self.client().post("/v1/expert/forecast", json=self.body())
        self.assertEqual(response.status_code, 200)
        expected = runtime.forecast(self.model, anchors_of(self.raw[self.raw["stay_id"] == 2]))
        self.assertEqual(response.json()["forecasts"], json.loads(json.dumps(expected["forecasts"])))
        self.assertFalse(response.json()["boundary"]["clinical_promotion_allowed"])

    def test_bad_requests_are_rejected(self):
        client = self.client()
        body = self.body()
        body["anchors"][0]["values"]["potassium_t"] = "high"
        self.assertEqual(client.post("/v1/expert/forecast", json=body).status_code, 422)
        body = self.body()
        body["anchors"][1]["time"] = body["anchors"][0]["time"]
        self.assertEqual(client.post("/v1/expert/forecast", json=body).status_code, 422)
        self.assertEqual(client.post("/v1/expert/forecast", json={"anchors": []}).status_code, 422)

    def test_cross_origin_calls_need_an_allowed_origin(self):
        response = self.client().options("/v1/expert/forecast", headers={
            "Origin": "https://elsewhere.example", "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_cells_endpoint_lists_inputs(self):
        listing = self.client().get("/v1/expert/cells").json()
        self.assertEqual([c["cell"] for c in listing["cells"]], [CELL])
        self.assertIn("potassium_t", listing["input_columns"])


if __name__ == "__main__":
    unittest.main()

"""Forecast runtime adapter contract: the 45-cell catalog, statuses, detached mode and fail-closed checks.

The attached mode is exercised against a stub of the forecast API served on a
local port, so these tests need no model files.
"""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest

from oslian_demo.monitoring import anatomy
from oslian_demo.monitoring.forecast_runtime import (
    INTERVAL_STATUS, REASON_NOT_ATTACHED, REASON_OUTSIDE_EPISODE, ForecastRuntime, RuntimeConfig, RuntimeUnavailable,
    load_contract_fixture,
)
from oslian_demo.monitoring.timeline import build_model_input

DETACHED = RuntimeConfig("detached", "http://127.0.0.1:1")
ONSET = "2026-03-14T06:00:00.000Z"
ROWS = [
    {"effective_time": "2026-03-14T10:00:00.000Z", "variable": "heart_rate", "value": 88.0, "resource_id": "hr", "version_id": "1"},
    {"effective_time": "2026-03-14T12:00:00.000Z", "variable": "creatinine", "value": 1.5, "resource_id": "cr", "version_id": "1"},
]
STATE = {"heart_rate": {"value": 88.0, "age_hours": 2.0, "count": 1}, "creatinine": {"value": 1.5, "age_hours": 0.0, "count": 1}}


class StubApi:
    """The two endpoints the runtime calls, answering from the pinned contract unless a test changes it."""

    def __init__(self):
        self.contract = load_contract_fixture()
        self.registry_sha256 = self.contract["registry_sha256"]
        self.cells = [{"cell": c["cell"], "target": c["target"], "horizon_hours": c["horizon_hours"],
                       "recipe": c["source_model"], "flags": c["flags"]} for c in self.contract["cells"]]
        self.forecast_status = 200
        self.edit = lambda reply: reply
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"cells": stub.cells, "model": {"registry_sha256": stub.registry_sha256}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                if stub.forecast_status != 200:
                    return self._send(stub.forecast_status, {"detail": "refused"})
                last = body["anchors"][-1]
                forecasts = []
                for cell in stub.cells:
                    current = last["values"].get(f"{cell['target']}_t")
                    forecasts.append({"cell": cell["cell"], "status": "ok" if current is not None else "current_value_missing",
                                      "point": current + 1.0 if current is not None else None})
                self._send(200, stub.edit({"anchor_time": last["time"].replace(".000Z", ""), "forecasts": forecasts,
                                           "model": {"registry_sha256": stub.registry_sha256}}))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def model_input(cutoff="2026-03-14T12:00:00.000Z"):
    return build_model_input([r for r in ROWS if r["effective_time"] <= cutoff], cutoff, ONSET)


class ContractFixtureTests(unittest.TestCase):
    def test_the_contract_pins_the_registry_and_the_45_cells(self):
        contract = load_contract_fixture()
        self.assertEqual(contract["schema"], "oslian.forecast-api-contract.v1")
        self.assertEqual(len(contract["registry_sha256"]), 64)
        self.assertEqual(len(contract["api_commit"]), 40)
        self.assertEqual(contract["anchor_spacing_hours"], 6.0)
        self.assertFalse(contract["boundary"]["causal_claim_allowed"])
        self.assertFalse(contract["boundary"]["clinical_promotion_allowed"])
        cells = contract["cells"]
        self.assertEqual(len(cells), 45)
        for cell in cells:
            self.assertEqual(set(cell), {"cell", "target", "horizon_hours", "unit", "source_model", "flags"})
            self.assertEqual(cell["cell"], f"{cell['target']}@{cell['horizon_hours']}h")


class DetachedRuntimeTests(unittest.TestCase):
    def test_default_mode_is_detached_and_degraded(self):
        self.assertEqual(RuntimeConfig.from_env({}).mode, "detached")
        runtime = ForecastRuntime(DETACHED)
        self.assertFalse(runtime.attached())
        result = runtime.run(model_input(), STATE)
        self.assertEqual(result.summary, {"catalog_cells": 45, "model_cells": 0, "missing_cells": 0, "degraded_cells": 45})
        for cell in result.cells:
            self.assertEqual(cell["status"], "degraded")
            self.assertEqual(cell["fallback_reason"], REASON_NOT_ATTACHED)
            self.assertIsNone(cell["forecast_point"])
            self.assertIsNone(cell["lower"])
            self.assertIsNone(cell["upper"])
            self.assertFalse(cell["causal_claim_allowed"])
        observed = next(c for c in result.cells if c["cell"] == "creatinine@6h")
        self.assertEqual(observed["observed_value"], 1.5, "the observation is still shown; only the forecast is withheld")
        health = runtime.health()
        self.assertEqual(health["status"], "runtime_not_attached")
        self.assertFalse(health["ready"])

    def test_capabilities_come_from_the_contract(self):
        capabilities = ForecastRuntime(DETACHED).capabilities()
        self.assertEqual(capabilities["catalog_cells"], 45)
        self.assertEqual(capabilities["interval_cells"], 0)
        self.assertEqual(capabilities["anchor_spacing_hours"], 6.0)
        self.assertFalse(capabilities["causal_claim_allowed"])
        self.assertFalse(capabilities["clinical_promotion_allowed"])
        self.assertFalse(capabilities["clinical_alerts_supported"])
        self.assertIn("creatinine@6h", {c["cell"] for c in capabilities["cells"]})

    def test_env_config_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            RuntimeConfig.from_env({"OSLIAN_FORECAST_RUNTIME": "production"})


class AttachedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.api = StubApi()
        self.runtime = ForecastRuntime(RuntimeConfig("api", self.api.url))

    def tearDown(self):
        self.api.close()

    def test_a_run_sends_the_anchor_stream_and_normalizes_every_cell(self):
        result = self.runtime.run(model_input(), STATE)
        self.assertTrue(self.runtime.attached())
        anchors = self.api.requests[-1]["anchors"]
        self.assertEqual([a["time"] for a in anchors], ["2026-03-14T12:00:00.000Z"])
        self.assertEqual(anchors[0]["values"], {"creatinine_t": 1.5, "creatinine_age_hr": 0.0,
                                                "heart_rate_t": 88.0, "heart_rate_age_hr": 2.0})
        self.assertEqual(len(result.cells), 45)
        cell = next(c for c in result.cells if c["cell"] == "creatinine@6h")
        self.assertEqual((cell["status"], cell["forecast_point"], cell["unit"]), ("model", 2.5, "mg/dL"))
        self.assertEqual(cell["interval_status"], INTERVAL_STATUS)
        self.assertIsNone(cell["lower"])
        missing = next(c for c in result.cells if c["cell"] == "map@6h")
        self.assertEqual((missing["status"], missing["forecast_point"], missing["fallback_reason"]),
                         ("missing", None, "current_value_missing"))
        self.assertEqual(result.summary["model_cells"] + result.summary["missing_cells"], 45)
        self.assertEqual(result.summary["degraded_cells"], 0)

    def test_a_value_older_than_the_lookback_is_missing_at_the_anchor(self):
        self.runtime.run(model_input("2026-03-14T18:00:00.000Z"), STATE)
        first, second = self.api.requests[-1]["anchors"]
        self.assertIn("heart_rate_t", first["values"])
        self.assertNotIn("heart_rate_t", second["values"], "measured 8 h before the 12 h anchor")
        self.assertEqual(second["values"], {"creatinine_t": 1.5, "creatinine_age_hr": 6.0})

    def test_past_the_fitted_episode_window_no_forecast_is_requested(self):
        result = self.runtime.run(model_input("2026-03-17T12:00:00.000Z"), STATE)      # onset + 78 h
        self.assertEqual(self.api.requests, [])
        self.assertEqual({c["fallback_reason"] for c in result.cells}, {REASON_OUTSIDE_EPISODE})
        self.assertEqual(result.summary["degraded_cells"], 45)

    def test_another_registry_fails_closed(self):
        self.api.registry_sha256 = "0" * 64
        with self.assertRaises(RuntimeUnavailable):
            self.runtime.run(model_input(), STATE)
        self.assertFalse(self.runtime.attached())
        health = self.runtime.health()
        self.assertEqual((health["status"], health["ready"]), ("degraded", False))

    def test_a_different_cell_list_fails_closed(self):
        self.api.cells = self.api.cells[:-1]
        with self.assertRaises(RuntimeUnavailable):
            self.runtime.run(model_input(), STATE)

    def test_a_reply_for_another_anchor_or_with_a_changed_registry_fails_closed(self):
        self.runtime.verify()
        for change in ({"anchor_time": "2026-03-14T06:00:00"}, {"model": {"registry_sha256": "0" * 64}}):
            self.api.edit = lambda reply, change=change: {**reply, **change}
            with self.assertRaises(RuntimeUnavailable):
                self.runtime.run(model_input(), STATE)

    def test_missing_cells_or_an_unknown_status_fail_closed(self):
        self.runtime.verify()
        self.api.edit = lambda reply: {**reply, "forecasts": reply["forecasts"][:-1]}
        with self.assertRaises(RuntimeUnavailable):
            self.runtime.run(model_input(), STATE)

        def unknown(reply):
            reply = copy.deepcopy(reply)
            reply["forecasts"][0]["status"] = "extrapolated"
            return reply
        self.api.edit = unknown
        with self.assertRaises(RuntimeUnavailable):
            self.runtime.run(model_input(), STATE)

    def test_a_refusal_or_an_unreachable_api_fails_closed(self):
        self.runtime.verify()
        self.api.forecast_status = 422
        with self.assertRaises(RuntimeUnavailable):
            self.runtime.run(model_input(), STATE)
        with self.assertRaises(RuntimeUnavailable):
            ForecastRuntime(RuntimeConfig("api", "http://127.0.0.1:1")).run(model_input(), STATE)


class AnatomyTests(unittest.TestCase):
    def test_organ_map_is_anatomical_and_cells_join_the_catalog(self):
        contract = load_contract_fixture()
        document = anatomy.anatomy_document(contract["cells"])
        ids = [o["id"] for o in document["organs"]]
        self.assertEqual(set(ids), set(anatomy.CANONICAL_ORGANS))
        for organ in document["organs"]:
            for word in anatomy.FUNCTIONAL_SYSTEM_WORDS:
                self.assertNotIn(word, organ["id"])
        kidneys = next(o for o in document["organs"] if o["id"] == "kidneys")
        self.assertIn("creatinine@6h", {c["cell"] for c in kidneys["cells"]})
        lungs = next(o for o in document["organs"] if o["id"] == "lungs")
        self.assertIn("o2sat@12h", {c["cell"] for c in lungs["cells"]})
        brain = next(o for o in document["organs"] if o["id"] == "brain")
        self.assertEqual(brain["cells"], [])
        self.assertEqual(document["unattributed_variables"], [])
        total = sum(len(o["cells"]) for o in document["organs"]) + len(document["systemic"]["cells"])
        self.assertGreaterEqual(total, len(contract["cells"]), "every cell reaches at least one organ or systemic")

    def test_functional_system_names_are_rejected_as_organs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema": "oslian.organ-variable-map.v1", "organs": [
                {"id": "acid_base_electrolyte", "label": "x", "variables": []}], "systemic": {"label": "s", "variables": []}}),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                anatomy.load_organ_map(path)


if __name__ == "__main__":
    unittest.main()

"""Past forecasts are scored against the measurement the cells were fitted against, and only once it is final."""

import os
import unittest

os.environ["OSLIAN_STREAM_LOG"] = "off"

from flask import Flask  # noqa: E402

from oslian_demo.monitoring.api import StreamService, create_blueprint  # noqa: E402
from oslian_demo.monitoring.forecast_runtime import ForecastRuntime, RuntimeConfig  # noqa: E402
from oslian_demo.monitoring.scoring import label_tolerance_hours, score_forecasts  # noqa: E402
from test_stream_runtime_contract import StubApi  # noqa: E402

T = lambda hour, minute=0: f"2026-03-14T{hour:02d}:{minute:02d}:00.000Z"


class FakeRepo:
    """Just what ``score_forecasts`` reads: runs, their cells and the delivered observations."""

    def __init__(self, runs, cells, observations):
        self.runs, self.cells, self.observations = runs, cells, observations

    def runs_for_patient(self, patient_id, limit=200):
        return list(self.runs)                                   # newest first, as the repository returns them

    def cells_for_run(self, run_id):
        return list(self.cells[run_id])

    def delivered_observations(self, session_id, epoch, cutoff=None):
        return list(self.observations)


def run(run_id, cutoff, status="completed"):
    return {"run_id": run_id, "session_id": "s", "epoch": 1, "status": status, "input_cutoff": cutoff}


def cell(target, horizon, point, observed, status="model"):
    return {"cell": f"{target}@{horizon}h", "target": target, "unit": "u", "horizon_hours": horizon,
            "forecast_point": point, "observed_value": observed, "status": status}


def obs(variable, time, value, rid=None):
    return {"variable": variable, "effective_time": time, "value": value, "resource_id": rid or f"{variable}-{time}", "version_id": "1"}


def score(runs, cells, observations):
    return score_forecasts(FakeRepo(runs, cells, observations), "s", 1, "p")


class LabelRuleTests(unittest.TestCase):
    def test_the_tolerance_is_the_fitted_rows_rule(self):
        self.assertEqual([label_tolerance_hours(h) for h in (1, 3, 6, 12, 48)], [0.5, 1.5, 2.0, 2.0, 2.0])
        self.assertEqual(label_tolerance_hours(0.25), 0.25)


class ScoringTests(unittest.TestCase):
    def test_a_resolved_forecast_reports_both_errors_and_its_source(self):
        result = score([run("r", T(12))], {"r": [cell("map", 6, 75.0, 70.0)]},
                       [obs("map", T(17, 30), 72.0, "near"), obs("map", T(19), 80.0, "far"), obs("map", T(21), 60.0)])
        item = result["forecasts"][0]
        self.assertEqual((item["state"], item["measured_value"], item["reference"]), ("resolved", 72.0, "Observation/near/_history/1"))
        self.assertEqual((item["forecast_error"], item["carried_forward_error"]), (3.0, -2.0))
        self.assertEqual(item["forecast_time"], T(18))
        summary = result["cells"][0]
        self.assertEqual((summary["resolved"], summary["compared"]), (1, 1))
        self.assertEqual((summary["mean_abs_error_forecast"], summary["mean_abs_error_carried_forward"]), (3.0, 2.0))

    def test_it_stays_pending_until_no_nearer_measurement_can_arrive(self):
        result = score([run("r", T(12))], {"r": [cell("map", 6, 75.0, 70.0)]}, [obs("map", T(17, 30), 72.0), obs("hr", T(19, 59), 1.0)])
        self.assertEqual(result["forecasts"][0]["state"], "pending", "the window runs to 20:00 and the stream is at 19:59")
        self.assertEqual(result["counts"], {"resolved": 0, "pending": 1, "unmeasured": 0})
        self.assertIsNone(result["cells"][0]["mean_abs_error_forecast"])

    def test_a_closed_window_without_a_measurement_is_unmeasured(self):
        result = score([run("r", T(12))], {"r": [cell("map", 6, 75.0, 70.0)]}, [obs("map", T(15), 72.0), obs("map", T(21), 60.0)])
        self.assertEqual(result["forecasts"][0]["state"], "unmeasured", "15:00 and 21:00 are both more than 2 h from 18:00")

    def test_a_short_horizon_never_scores_against_the_anchors_own_value(self):
        result = score([run("r", T(12))], {"r": [cell("map", 1, 75.0, 70.0)]}, [obs("map", T(12), 70.0), obs("map", T(12, 40), 71.0), obs("map", T(14), 1.0)])
        item = result["forecasts"][0]
        self.assertEqual((item["state"], item["measured_value"]), ("resolved", 71.0), "12:40 is within 0.5 h of 13:00 and after the anchor")
        alone = score([run("r", T(12))], {"r": [cell("map", 1, 75.0, 70.0)]}, [obs("map", T(12), 70.0), obs("map", T(14), 1.0)])
        self.assertEqual(alone["forecasts"][0]["state"], "unmeasured")

    def test_only_the_newest_completed_run_of_an_anchor_is_scored(self):
        runs = [run("failed", T(12), status="failed"), run("new", T(12)), run("old", T(12))]            # newest first
        cells = {"new": [cell("map", 6, 75.0, 70.0)], "old": [cell("map", 6, 99.0, 70.0)], "failed": []}
        result = score(runs, cells, [obs("map", T(18), 72.0), obs("map", T(21), 60.0)])
        self.assertEqual([(item["run_id"], item["forecast_point"]) for item in result["forecasts"]], [("new", 75.0)])
        self.assertEqual(result["anchors_scored"], 1)

    def test_cells_without_a_model_forecast_are_not_scored(self):
        cells = {"r": [cell("map", 6, None, 70.0, status="missing"), cell("hr", 6, None, 80.0, status="degraded")]}
        result = score([run("r", T(12))], cells, [obs("map", T(18), 72.0), obs("map", T(21), 60.0)])
        self.assertEqual((result["forecasts"], result["cells"]), ([], []))
        self.assertFalse(result["causal_claim_allowed"])
        self.assertIn("this stream only", result["scope"])


class ScoreRouteTests(unittest.TestCase):
    def setUp(self):
        self.api = StubApi()
        self.service = StreamService(":memory:", runtime=ForecastRuntime(RuntimeConfig("api", self.api.url)))
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(self.service))
        self.client = app.test_client()
        state = self.client.post("/api/demo-streams", json={"source": "synthetic"}).get_json()
        self.sid, self.pid = state["session_id"], state["patient_id"]

    def tearDown(self):
        self.service.close()
        self.api.close()

    def scores(self, steps):
        """Step the stream, letting the run of each clinical time finish before the next time is delivered.

        Stepping without a pause merges every request into one run of the newest anchor, as the scheduler is meant
        to; a stream played in time runs each anchor as it is reached, and that is what is scored.
        """
        session = self.service.sessions.get(self.sid)
        times = [e.resource.get("effectiveDateTime") or e.resource["effectivePeriod"]["end"] for e in session.events]
        for index in range(steps):
            if index and times[index] != times[index - 1]:
                session.scheduler.wait_idle(30)
            self.client.post(f"/api/demo-streams/{self.sid}/step")
        session.scheduler.wait_idle(30)
        return self.client.get(f"/api/patients/{self.pid}/forecast-scores").get_json()

    def test_nothing_is_resolved_at_the_first_anchor(self):
        document = self.scores(13)
        self.assertEqual(document["as_of"], T(12))
        self.assertGreater(document["counts"]["pending"], 0)
        self.assertEqual(document["counts"]["resolved"], 0)

    def test_the_whole_stream_scores_each_anchor_once_against_later_measurements(self):
        document = self.scores(147)
        self.assertEqual(document["schema"], "oslian.forecast-scores.v1")
        self.assertGreater(document["counts"]["resolved"], 20)
        keys = [(item["cell"], item["anchor_time"]) for item in document["forecasts"]]
        self.assertEqual(len(keys), len(set(keys)), "an anchor that was re-run is still one forecast per cell")
        for item in document["forecasts"]:
            if item["state"] != "resolved":
                continue
            self.assertGreater(item["measured_time"], item["anchor_time"])
            # the stub forecasts the anchor's value plus one, so the two errors differ by exactly one
            self.assertAlmostEqual(item["forecast_error"] - item["carried_forward_error"], 1.0)
        self.assertEqual(sum(cell["resolved"] for cell in document["cells"]), document["counts"]["resolved"])
        self.assertEqual(self.client.get("/api/patients/nobody/forecast-scores").status_code, 404)


if __name__ == "__main__":
    unittest.main()

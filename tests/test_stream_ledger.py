"""Append-only ledger, versioning, ordering, no-future-leak, and restart recovery."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from oslian_demo.monitoring.fhir import now_iso
from oslian_demo.monitoring.ingest import Ingestor
from oslian_demo.monitoring.repository import Repository
from oslian_demo.monitoring.source import MockFhirSource
from oslian_demo.monitoring.timeline import (build_model_input, delivered_timeline, organ_state,
                                  organ_state_for_run, run_timeline, series, variable_state)

SOURCE = "synthetic-fhir-r4"


def _session(repo, session_id="s1"):
    repo.create_session({
        "session_id": session_id, "patient_id": "syn-pat-0001", "source_system": SOURCE,
        "fixture_version": "synthetic_patient_v1", "status": "idle", "speed": 1.0, "epoch": 1, "cursor": 0,
        "sim_elapsed_seconds": 0.0, "last_event_sequence": 0, "created_at": now_iso(), "updated_at": now_iso()})


class HistoricalOrganSnapshotTests(unittest.TestCase):
    """A historical run's organ state is bounded by the observations that run consumed."""

    def setUp(self):
        self.repo = Repository(":memory:")
        self.source = MockFhirSource()
        self.ingestor = Ingestor(self.repo, SOURCE)
        self.ingestor.ingest_bundle(self.source.initial_bundle(), received_at=now_iso(), event_id="boot")
        _session(self.repo)
        self.events = {e.resource["id"] + "@" + e.resource["meta"]["versionId"]: e for e in self.source.events()}
        self.sequence = 0

    def deliver(self, resource):
        self.sequence += 1
        result = self.ingestor.ingest_resource(resource, received_at=now_iso(), event_id=f"e{self.sequence}")
        self.assertTrue(result.accepted, result.validation_issues)
        self.repo.record_delivery("s1", 1, result.raw_id, self.sequence, f"e{self.sequence}", now_iso())
        return result

    def make_run(self, run_id):
        """Record a run over everything delivered so far, exactly as the session does."""
        timeline = delivered_timeline(self.repo, "s1", 1)
        cutoff = max(row["effective_time"] for row in timeline)
        model_input = build_model_input(timeline, cutoff, "2026-03-14T06:00:00.000Z")
        self.repo.create_run({
            "run_id": run_id, "session_id": "s1", "epoch": 1, "patient_id": "syn-pat-0001", "status": "running",
            "runtime_mode": "fixture", "model_commit": None, "registry_sha256": None, "release_date": None,
            "input_cutoff": cutoff, "input_watermark": model_input.watermark,
            "input_resource_count": len(model_input.resource_refs), "input_hash": model_input.input_hash,
            "input_refs_json": json.dumps(list(model_input.resource_refs)),
            "trigger_event_ids_json": "{}", "started_at": now_iso()})
        return self.repo.run(run_id)

    def test_a_run_snapshot_covers_the_organs_of_the_observations_it_used(self):
        self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource)
        self.deliver(self.events["syn-obs-creatinine-h06@1"].resource)
        run = self.make_run("run-a")
        organs = organ_state_for_run(self.repo, run)
        self.assertIn("heart", organs)
        self.assertIn("kidneys", organs)
        self.assertIn("heart_rate", organs["heart"]["variables"])
        self.assertIn("creatinine", organs["kidneys"]["variables"])
        for item in organs.values():
            self.assertGreater(item["count"], 0)
            self.assertIsNotNone(item["at"])
            self.assertLessEqual(item["at"], run["input_cutoff"])
            self.assertGreaterEqual(item["sequence"], 1)
            self.assertGreaterEqual(item["age_hours"], 0)

    def test_later_deliveries_never_reach_an_earlier_run(self):
        self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource)
        early = self.make_run("run-early")
        before = organ_state_for_run(self.repo, early)
        self.assertNotIn("kidneys", before)

        self.deliver(self.events["syn-obs-creatinine-h06@1"].resource)
        late = self.make_run("run-late")
        self.assertEqual(organ_state_for_run(self.repo, early), before, "an earlier run must not move")
        self.assertIn("kidneys", organ_state_for_run(self.repo, late))

    def test_a_late_arrival_with_an_early_effective_time_does_not_leak_backwards(self):
        """The classic causal leak: delivered after the run, but timestamped before its cutoff."""
        self.deliver(self.events["syn-obs-heart_rate-h20@1"].resource)
        run = self.make_run("run-x")
        before = organ_state_for_run(self.repo, run)
        self.assertNotIn("lungs", before)

        late = self.events["syn-obs-o2sat-h12@1"].resource      # effective time well before the cutoff
        self.assertLess(late["effectiveDateTime"], run["input_cutoff"])
        self.deliver(late)

        self.assertEqual(organ_state_for_run(self.repo, run), before,
                         "an observation the run never saw must not appear in its snapshot")
        # and the naive alternative - re-filtering the current timeline by cutoff - would have leaked it
        naive = organ_state(delivered_timeline(self.repo, "s1", 1, run["input_cutoff"]), run["input_cutoff"])
        self.assertIn("lungs", naive)

    def test_the_historical_timeline_signals_and_charts_are_frozen_too(self):
        """Whatever freezes the organs must freeze the series and the variable state."""
        self.deliver(self.events["syn-obs-heart_rate-h20@1"].resource)
        self.deliver(self.events["syn-obs-creatinine-h06@1"].resource)
        run = self.make_run("run-frozen")
        rows = run_timeline(self.repo, run)
        before = {
            "rows": [(r["resource_id"], r["version_id"], r["value"]) for r in rows],
            "state": variable_state(rows, run["input_cutoff"]),
            "series": {v: series(rows, v) for v in sorted({r["variable"] for r in rows})},
            "organs": organ_state_for_run(self.repo, run),
        }
        self.assertIn("creatinine", before["state"])

        self.deliver(self.events["syn-obs-o2sat-h12@1"].resource)          # late, but timestamped earlier
        self.deliver(self.events["syn-obs-temperature-h26@1"].resource)    # after the cutoff

        rows = run_timeline(self.repo, run)
        after = {
            "rows": [(r["resource_id"], r["version_id"], r["value"]) for r in rows],
            "state": variable_state(rows, run["input_cutoff"]),
            "series": {v: series(rows, v) for v in sorted({r["variable"] for r in rows})},
            "organs": organ_state_for_run(self.repo, run),
        }
        self.assertEqual(after, before, "a historical anchor must not move under new deliveries")
        self.assertNotIn("o2sat", after["state"])

    def test_the_historical_timeline_and_the_organ_snapshot_describe_one_set(self):
        """The figure and the charts read the same rows, so they cannot disagree."""
        from oslian_demo.monitoring.routing import variable_routing
        for name in ("syn-obs-heart_rate-h02@1", "syn-obs-creatinine-h06@1", "syn-obs-o2sat-h12@1"):
            self.deliver(self.events[name].resource)
        run = self.make_run("run-one-set")
        rows = run_timeline(self.repo, run)
        routing = variable_routing()
        expected = {o for r in rows for o in routing.organs_for(r["variable"])}
        self.assertEqual(set(organ_state_for_run(self.repo, run)), expected)
        for organ, item in organ_state_for_run(self.repo, run).items():
            for variable in item["variables"]:
                self.assertIn(variable, variable_state(rows, run["input_cutoff"]), f"{organ}/{variable}")

    def test_an_amendment_after_the_run_does_not_rewrite_its_snapshot(self):
        self.deliver(self.events["syn-obs-temperature-h26@1"].resource)
        self.deliver(self.events["syn-obs-heart_rate-h20@1"].resource)
        run = self.make_run("run-v1")
        before = organ_state_for_run(self.repo, run)
        self.deliver(self.events["syn-obs-temperature-h26@2"].resource)   # entered-in-error amendment
        self.assertEqual(organ_state_for_run(self.repo, run), before)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repository(":memory:")
        self.source = MockFhirSource()
        self.ingestor = Ingestor(self.repo, SOURCE)
        self.ingestor.ingest_bundle(self.source.initial_bundle(), received_at=now_iso(), event_id="boot")
        _session(self.repo)
        self.events = {e.resource["id"] + "@" + e.resource["meta"]["versionId"]: e for e in self.source.events()}
        self.sequence = 0

    def deliver(self, resource, epoch=1):
        self.sequence += 1
        result = self.ingestor.ingest_resource(resource, received_at=now_iso(), event_id=f"e{self.sequence}")
        self.assertTrue(result.accepted, result.validation_issues)
        new = self.repo.record_delivery("s1", epoch, result.raw_id, self.sequence, f"e{self.sequence}", now_iso())
        return result, new

    def test_duplicate_delivery_is_idempotent(self):
        hr = self.events["syn-obs-heart_rate-h10@1"].resource
        before = self.repo.raw_count()
        first, new1 = self.deliver(hr)
        second, new2 = self.deliver(copy.deepcopy(hr))
        self.assertEqual(first.raw_id, second.raw_id)
        self.assertFalse(first.ledger_duplicate)
        self.assertTrue(second.ledger_duplicate)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(self.repo.raw_count(), before + 1)
        self.assertEqual(len(self.repo.canonical_for_raw(first.raw_id)), 1)

    def test_amended_version_supersedes_without_overwriting(self):
        v1 = self.events["syn-obs-potassium-h12@1"].resource
        v2 = self.events["syn-obs-potassium-h12@2"].resource
        self.deliver(v1)
        timeline = delivered_timeline(self.repo, "s1", 1)
        k = [r for r in timeline if r["resource_id"] == "syn-obs-potassium-h12"]
        self.assertEqual(len(k), 1)
        self.assertEqual(k[0]["version_id"], "1")
        self.assertAlmostEqual(k[0]["value"], 5.9)
        self.deliver(v2)
        timeline = delivered_timeline(self.repo, "s1", 1)
        k = [r for r in timeline if r["resource_id"] == "syn-obs-potassium-h12"]
        self.assertEqual(len(k), 1)
        self.assertEqual(k[0]["version_id"], "2")
        self.assertEqual(k[0]["observation_status"], "amended")
        self.assertAlmostEqual(k[0]["value"], 4.4)
        self.assertIsNotNone(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-potassium-h12", "1"))
        self.assertIsNotNone(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-potassium-h12", "2"))

    def test_entered_in_error_removes_resource_from_timeline(self):
        self.deliver(self.events["syn-obs-temperature-h26@1"].resource)
        self.assertTrue(any(r["resource_id"] == "syn-obs-temperature-h26" for r in delivered_timeline(self.repo, "s1", 1)))
        result, _ = self.deliver(self.events["syn-obs-temperature-h26@2"].resource)
        self.assertEqual(result.mapping_failure, "excluded_status")
        self.assertFalse(any(r["resource_id"] == "syn-obs-temperature-h26" for r in delivered_timeline(self.repo, "s1", 1)))

    def test_out_of_order_arrival_is_placed_by_clinical_time(self):
        late = self.events["syn-obs-lactate-h18@1"].resource        # effective 18 h, arrives after 20 h vitals
        self.deliver(self.events["syn-obs-heart_rate-h20@1"].resource)
        self.deliver(late)
        timeline = delivered_timeline(self.repo, "s1", 1)
        order = [r["resource_id"] for r in timeline if r["resource_id"] in {"syn-obs-lactate-h18", "syn-obs-heart_rate-h20"}]
        self.assertEqual(order, ["syn-obs-lactate-h18", "syn-obs-heart_rate-h20"])
        model_input = build_model_input(timeline, "2026-03-15T02:00:00.000Z", "2026-03-14T06:00:00.000Z")
        self.assertIn("18.000000,lactate", model_input.csv_text)
        self.assertIn("Observation/syn-obs-lactate-h18/_history/1", model_input.resource_refs)

    def test_no_future_leak_past_input_cutoff(self):
        self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource)
        self.deliver(self.events["syn-obs-heart_rate-h04@1"].resource)
        cutoff = "2026-03-14T08:00:00.000Z"
        timeline = delivered_timeline(self.repo, "s1", 1, cutoff)
        self.assertTrue(all(r["effective_time"] <= cutoff for r in timeline))
        self.assertFalse(any(r["resource_id"] == "syn-obs-heart_rate-h04" for r in timeline))
        model_input = build_model_input(timeline, cutoff, "2026-03-14T06:00:00.000Z")
        self.assertNotIn("4.000000,heart_rate", model_input.csv_text)
        self.assertEqual(model_input.watermark, cutoff)
        with self.assertRaises(ValueError):
            build_model_input(delivered_timeline(self.repo, "s1", 1), cutoff, "2026-03-14T06:00:00.000Z")

    def test_reset_epoch_keeps_ledger_and_replays_without_new_rows(self):
        self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource)
        before = self.repo.raw_count()
        result, new = self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource, epoch=2)
        self.assertTrue(result.ledger_duplicate)
        self.assertTrue(new, "a new epoch re-delivers the same ledger row")
        self.assertEqual(self.repo.raw_count(), before)
        self.assertEqual(len(delivered_timeline(self.repo, "s1", 2)), 1)

    def test_variable_state_ages_relative_to_clinical_now(self):
        self.deliver(self.events["syn-obs-heart_rate-h02@1"].resource)
        state = variable_state(delivered_timeline(self.repo, "s1", 1), "2026-03-14T10:00:00.000Z")
        self.assertAlmostEqual(state["heart_rate"]["age_hours"], 2.0)
        self.assertEqual(state["heart_rate"]["count"], 1)


class RestartRecoveryTests(unittest.TestCase):
    def test_sessions_runs_and_events_survive_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stream.sqlite3"
            repo = Repository(path)
            _session(repo)
            repo.update_session("s1", status="running", updated_at=now_iso())
            repo.append_event("s1", "ev1", "stream.state", {"status": "running"}, now_iso())
            repo.create_run({
                "run_id": "r1", "session_id": "s1", "epoch": 1, "patient_id": "syn-pat-0001", "status": "running",
                "runtime_mode": "fixture", "model_commit": "abc", "registry_sha256": "def", "release_date": "2026-08-22",
                "input_cutoff": "2026-03-14T08:00:00.000Z", "input_watermark": "2026-03-14T08:00:00.000Z",
                "input_resource_count": 1, "input_hash": "h", "input_refs_json": json.dumps(["Observation/x/_history/1"]),
                "trigger_event_ids_json": "{}", "started_at": now_iso()})
            repo.finish_run("r1", status="completed", completed_at=now_iso(), duration_ms=1, error=None, summary={"catalog_cells": 41})
            repo.insert_cells("r1", [{"cell": "map@1h", "target": "map", "horizon_hours": 1, "unit": "mmHg", "observed_value": 70.0,
                                       "measurement_age_hours": 0.0, "observations": 1, "forecast_point": None,
                                       "source_model": "none", "status": "degraded", "fallback_reason": "x",
                                       "interval_status": "none", "lower": None, "upper": None, "causal_claim_allowed": False}])
            repo.close()

            from oslian_demo.monitoring.session import SessionManager
            from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime
            reopened = Repository(path)
            manager = SessionManager(reopened, WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None)))
            session = manager.get("s1")
            self.assertIsNotNone(session)
            self.assertEqual(session.status, "paused", "a running session is recovered as paused after restart")
            self.assertEqual(reopened.events_after("s1", 0)[0]["event_type"], "stream.state")
            self.assertEqual(reopened.run("r1")["status"], "completed")
            self.assertEqual(reopened.cells_for_run("r1")[0]["cell"], "map@1h")
            manager.shutdown()
            reopened.close()


if __name__ == "__main__":
    unittest.main()

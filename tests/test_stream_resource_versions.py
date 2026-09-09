"""One resource id and version, one content: the ledger, the ingestor, the stream.

The ledger is append-only and keyed by (source system, resource type, id,
version). Delivering the same resource again is idempotent; delivering a
different resource under an identity the ledger already holds is a resource
version conflict and is refused, never served from the old copy. A fixture
revision that changes a resource therefore raises that resource's version, and a
session opened on an earlier revision is not resumed onto the new one: an old
ledger meets the revised stream as new versions, from the first event, and no
timeline mixes old content with new.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from oslian_demo.config import FIXTURE_DIR
from oslian_demo.monitoring import build_synthetic_fixture as builder
from oslian_demo.monitoring.fhir import now_iso
from oslian_demo.monitoring.ingest import Ingestor
from oslian_demo.monitoring.reference_ranges import DEMO_REFERENCE_RANGES
from oslian_demo.monitoring.repository import Repository, ResourceVersionConflict
from oslian_demo.monitoring.session import SessionManager
from oslian_demo.monitoring.source import MockFhirSource, make_source
from oslian_demo.monitoring.timeline import delivered_timeline
from oslian_demo.monitoring.wholebody import RuntimeConfig, WholeBodyRuntime

SOURCE = "synthetic-fhir-r4"
FIXTURE = FIXTURE_DIR / "synthetic_patient_v1.json"
SUPPLIED_CODES = {"2951-2", "2823-3", "2075-0", "1963-8", "3094-0", "2524-7", "6690-2", "777-3"}


def _runtime():
    return WholeBodyRuntime(RuntimeConfig("fixture", None, "python", None))


def _old_fixture_path(tmp: str) -> Path:
    """The stream as revision 1 shipped it: the same resources without a reference
    range or a reading, one version lower where revision 2 raised it."""
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for entry in document["initial_bundle"]["entry"]:
        _downgrade(entry["resource"])
    for event in document["events"]:
        _downgrade(event["resource"])
    document["fixture_version"] = "synthetic_patient_v1"
    document.pop("revision", None)
    document.pop("revision_notes", None)
    document.pop("reference_ranges", None)
    path = Path(tmp) / "synthetic_patient_r1.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _downgrade(resource):
    if resource.get("resourceType") != "Observation":
        return
    code = resource["code"]["coding"][0]["code"]
    if code in SUPPLIED_CODES:
        resource.pop("referenceRange", None)
        resource.pop("interpretation", None)
        # revision 2 raised every changed version past all that revision 1 had used:
        # by one, and by two for the potassium whose id had used 1 and 2
        taken = 2 if resource["id"] == "syn-obs-potassium-h12" else 1
        resource["meta"]["versionId"] = str(int(resource["meta"]["versionId"]) - taken)


def _resources(source):
    return [e["resource"] for e in source.initial_bundle()["entry"]] + [e.resource for e in source.events()]


def _observation(source, resource_id, version):
    return next(r for r in _resources(source) if r["id"] == resource_id and r["meta"]["versionId"] == version)


class LedgerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.repo = Repository(":memory:")
        self.source = MockFhirSource()
        self.bun = _observation(self.source, "syn-obs-bun-h00", "2")

    def _insert(self, resource, version=None):
        return self.repo.insert_raw(
            source_system=SOURCE, resource_type="Observation", resource_id=resource["id"],
            version_id=version or resource["meta"]["versionId"], patient_id="syn-pat-0001",
            source_last_updated=resource["meta"]["lastUpdated"], received_at=now_iso(), event_id="e",
            resource=resource, validation_status="valid")

    def test_the_same_resource_delivered_twice_is_idempotent(self):
        first = self._insert(self.bun)
        again = self._insert(copy.deepcopy(self.bun))
        self.assertTrue(first.created)
        self.assertFalse(again.created)
        self.assertEqual(first.raw_id, again.raw_id)
        self.assertEqual(self.repo.raw_count(), 1)

    def test_a_different_payload_under_the_same_identity_is_refused(self):
        self._insert(self.bun)
        changed = copy.deepcopy(self.bun)
        changed["valueQuantity"]["value"] = 99.5
        with self.assertRaises(ResourceVersionConflict) as caught:
            self._insert(changed)
        message = str(caught.exception)
        for part in (SOURCE, "Observation", "syn-obs-bun-h00", "version 2"):
            self.assertIn(part, message)
        # what a diagnosis needs, and not the patient's numbers
        self.assertNotIn("99.5", message)
        self.assertNotIn("valueQuantity", message)
        self.assertEqual((caught.exception.resource_id, caught.exception.version_id), ("syn-obs-bun-h00", "2"))
        # the ledger is exactly as it was
        self.assertEqual(self.repo.raw_count(), 1)
        self.assertEqual(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-bun-h00", "2")["resource"], self.bun)

    def test_a_new_version_is_accepted(self):
        self._insert(self.bun)
        newer = copy.deepcopy(self.bun)
        newer["meta"]["versionId"] = "3"
        newer["valueQuantity"]["value"] = 27.0
        self.assertTrue(self._insert(newer).created)
        self.assertEqual(self.repo.raw_count(), 2)

    def test_the_ingestor_refuses_a_conflict_and_says_which_identity(self):
        ingestor = Ingestor(self.repo, SOURCE)
        self.assertTrue(ingestor.ingest_resource(self.bun, received_at=now_iso(), event_id="e1").accepted)
        changed = copy.deepcopy(self.bun)
        changed["valueQuantity"]["value"] = 99.5
        result = ingestor.ingest_resource(changed, received_at=now_iso(), event_id="e2")
        self.assertFalse(result.accepted)
        self.assertTrue(result.version_conflict)
        self.assertIsNone(result.raw_id)
        self.assertIn("syn-obs-bun-h00", result.validation_issues[0])
        self.assertNotIn("99.5", result.validation_issues[0])
        self.assertEqual(self.repo.raw_count(), 1)


class FixtureRevisionTests(unittest.TestCase):
    """Revision 2 raised the version of every resource it changed, and only those."""

    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.resources = [e["resource"] for e in cls.document["initial_bundle"]["entry"]] + \
            [e["resource"] for e in cls.document["events"]]

    def test_the_document_names_its_revision(self):
        self.assertEqual(self.document["revision"], builder.FIXTURE_REVISION)
        self.assertEqual(self.document["fixture_version"], f"synthetic_patient_v1.r{builder.FIXTURE_REVISION}")
        self.assertTrue(any("r2" in note and "versionId" in note for note in self.document["revision_notes"]))

    def test_changed_resources_carry_a_raised_version_and_unchanged_ones_do_not(self):
        for resource in self.resources:
            if resource.get("resourceType") != "Observation":
                self.assertEqual(resource["meta"]["versionId"], "1", resource["id"])
                continue
            code = resource["code"]["coding"][0]["code"]
            version = int(resource["meta"]["versionId"])
            if code in SUPPLIED_CODES:
                self.assertGreaterEqual(version, 2, resource["id"])
                self.assertIn("referenceRange", resource)
            else:
                self.assertLessEqual(version, 2, resource["id"])   # only the retracted temperature reaches 2
                self.assertNotIn("referenceRange", resource)

    def test_no_identity_names_two_contents_and_the_amendment_order_holds(self):
        seen = {}
        for resource in self.resources:
            key = (resource["id"], resource["meta"]["versionId"])
            if key in seen:
                self.assertEqual(seen[key], resource, key)
            seen[key] = resource
        versions = {}
        for resource in self.resources:
            versions.setdefault(resource["id"], []).append(resource["meta"]["versionId"])
        self.assertEqual(versions["syn-obs-potassium-h12"], ["3", "4"])
        self.assertEqual(versions["syn-obs-temperature-h26"], ["1", "2"])
        self.assertEqual(versions["syn-obs-heart_rate-h10"], ["1", "1"])   # the duplicate delivery, unchanged


class StreamRevisionTests(unittest.TestCase):
    """The stream on an old ledger, on a new ledger, and across a reset and a restart."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.old_path = _old_fixture_path(self.tmp.name)
        self.repo = Repository(":memory:")
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.shutdown()
        self.repo.close()

    def _manager(self, factory=make_source):
        manager = SessionManager(self.repo, _runtime(), source_factory=factory)
        self.managers.append(manager)
        return manager

    def _old_factory(self, key="synthetic"):
        return MockFhirSource(self.old_path)

    def _play(self, session, steps):
        for _ in range(steps):
            session.step()

    def _rejections(self, session_id):
        return [e for e in self.repo.events_after(session_id, 0, limit=5000) if e["event_type"] == "fhir.resource.rejected"]

    def _normalized(self, session_id):
        return [e["payload"] for e in self.repo.events_after(session_id, 0, limit=5000)
                if e["event_type"] == "observation.normalized"]

    def _raw_of(self, row):
        return self.repo.raw(int(row["raw_id"]))["resource"]

    def test_an_amendment_still_supersedes_and_a_first_delivery_never_does(self):
        manager = self._manager()
        session, _ = manager.get_or_create("synthetic")
        self._play(session, 90)
        potassium = [p for p in self._normalized(session.session_id) if p["resource_id"] == "syn-obs-potassium-h12"]
        self.assertEqual([(p["version_id"], p["superseding"]) for p in potassium], [("3", False), ("4", True)])
        current = [r for r in delivered_timeline(self.repo, session.session_id, session.epoch)
                   if r["resource_id"] == "syn-obs-potassium-h12"]
        self.assertEqual((current[0]["version_id"], current[0]["value"]), ("4", 4.4))
        # a version 2 first delivery (revision 2 renumbering) is not an amendment
        first = next(p for p in self._normalized(session.session_id) if p["resource_id"] == "syn-obs-bun-h06")
        self.assertEqual((first["version_id"], first["superseding"]), ("2", False))

    def test_a_new_ledger_carries_the_range_from_the_first_observation_on(self):
        manager = self._manager()
        session, created = manager.get_or_create("synthetic")
        self.assertTrue(created)
        self._play(session, 147)
        self.assertEqual(self._rejections(session.session_id), [])
        for payload in self._normalized(session.session_id):
            for value in payload["values"]:
                if value["variable"] in DEMO_REFERENCE_RANGES:
                    self.assertTrue(payload["reference_range"], payload["resource_id"])
                    self.assertTrue(payload["interpretation"], payload["resource_id"])
        for row in delivered_timeline(self.repo, session.session_id, session.epoch):
            if row["variable"] in DEMO_REFERENCE_RANGES:
                self.assertIn("referenceRange", self._raw_of(row), row["resource_id"])

    def test_an_old_ledger_meets_the_revised_stream_without_a_mixed_timeline(self):
        # A ledger written by revision 1: a session that delivered thirty old events.
        old = self._manager(self._old_factory)
        old_session, _ = old.get_or_create("synthetic")
        self._play(old_session, 30)
        old_id = old_session.session_id
        old_cursor = self.repo.session(old_id)["cursor"]
        self.assertEqual(self.repo.session(old_id)["fixture_version"], "synthetic_patient_v1")
        old.shutdown()
        self.managers.remove(old)

        # The revised fixture opens the same ledger. The old session is not resumed
        # onto it; a new session starts from its first event and is all revision 2.
        new = self._manager()
        self.assertIsNone(new.get(old_id))
        self.assertEqual(self.repo.session(old_id)["cursor"], old_cursor)
        session, created = new.get_or_create("synthetic")
        self.assertTrue(created)
        self.assertNotEqual(session.session_id, old_id)
        self._play(session, 147)
        self.assertEqual(self._rejections(session.session_id), [])
        for row in delivered_timeline(self.repo, session.session_id, session.epoch):
            resource = self._raw_of(row)
            if row["variable"] in DEMO_REFERENCE_RANGES:
                self.assertIn("referenceRange", resource, row["resource_id"])
                self.assertGreaterEqual(int(row["version_id"]), 2, row["resource_id"])
            else:
                self.assertNotIn("referenceRange", resource, row["resource_id"])
        # the old session's own timeline is untouched: still revision 1, still without ranges
        for row in delivered_timeline(self.repo, old_id, 1):
            self.assertNotIn("referenceRange", self._raw_of(row), row["resource_id"])
        # both contents live side by side under their own versions
        self.assertIsNotNone(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-bun-h00", "1"))
        self.assertIsNotNone(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-bun-h00", "2"))

    def test_a_fixture_edited_without_a_version_bump_is_refused_at_the_ledger(self):
        # The honest stream has already written the whole ledger; a fixture that
        # changed a resource without raising its version now meets that ledger.
        honest = self._manager()
        honest_session, _ = honest.get_or_create("synthetic")
        self._play(honest_session, 147)
        honest.shutdown()
        self.managers.remove(honest)

        def tampered(key="synthetic"):
            source = make_source(key)
            bundle = json.loads(json.dumps(source.initial_bundle()))
            target = next(e["resource"] for e in bundle["entry"] if e["resource"]["id"] == "syn-obs-bun-h00")
            target["valueQuantity"]["value"] = 99.5
            source._bundle = bundle
            events = list(source.events())
            for event in events:
                if event.resource["id"] == "syn-obs-bun-h06":
                    event.resource["valueQuantity"]["value"] = 88.8
            source._events = tuple(events)
            return source

        # (a) a stored session on the same revision is resumed and reset; the tampered
        # event is refused when it arrives, named, and kept out of the timeline
        other = SessionManager(self.repo, _runtime(), source_factory=tampered)
        self.managers.append(other)
        session = other.get(honest_session.session_id)
        self.assertIsNotNone(session)
        session.reset()
        self._play(session, 40)
        refused = [e["payload"] for e in self._rejections(session.session_id) if e["payload"].get("version_conflict")]
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]["resource"]["id"], "syn-obs-bun-h06")
        self.assertIn("syn-obs-bun-h06", refused[0]["validation_issues"][0])
        self.assertNotIn("88.8", refused[0]["validation_issues"][0])
        bun = [r for r in delivered_timeline(self.repo, session.session_id, session.epoch) if r["resource_id"] == "syn-obs-bun-h06"]
        self.assertEqual(bun, [])
        self.assertEqual(self.repo.raw_by_reference(SOURCE, "Observation", "syn-obs-bun-h06", "2")["resource"]["valueQuantity"]["value"], 29.0)
        other.shutdown()
        self.managers.remove(other)

        # (b) with no session to resume, the tampered bootstrap bundle cannot open one
        for record in self.repo.sessions():
            self.repo.update_session(record["session_id"], fixture_version="elsewhere", updated_at=now_iso())
        fresh = SessionManager(self.repo, _runtime(), source_factory=tampered)
        self.managers.append(fresh)
        with self.assertRaises(ResourceVersionConflict) as caught:
            fresh.get_or_create("synthetic")
        self.assertEqual(caught.exception.resource_id, "syn-obs-bun-h00")
        self.assertNotIn("99.5", str(caught.exception))

    def test_reset_and_restart_keep_the_ranges_and_the_versions(self):
        manager = self._manager()
        session, _ = manager.get_or_create("synthetic")
        self._play(session, 40)
        session.reset()
        self._play(session, 20)
        self.assertEqual(self._rejections(session.session_id), [])
        rows = delivered_timeline(self.repo, session.session_id, session.epoch)
        self.assertTrue(rows)
        for row in rows:
            if row["variable"] in DEMO_REFERENCE_RANGES:
                self.assertIn("referenceRange", self._raw_of(row), row["resource_id"])
        before = [(r["resource_id"], r["version_id"], r["value"]) for r in rows]
        manager.shutdown()
        self.managers.remove(manager)
        # a restart resumes the session, on the same revision, with the same timeline
        again = self._manager()
        restored = again.get(session.session_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.epoch, session.epoch)
        after = [(r["resource_id"], r["version_id"], r["value"])
                 for r in delivered_timeline(self.repo, session.session_id, restored.epoch)]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

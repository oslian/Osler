"""Clinical data sources.

``ClinicalDataSource`` is the only seam between "where FHIR comes from" and
everything downstream. Both implementations hand the pipeline the same thing:
an initial R4 ``Bundle`` (type ``searchset``) and an ordered series of
``SourceEvent`` envelopes that wrap an unmodified R4 ``Observation``.

Downstream envelope (also the SSE ``fhir.resource.accepted`` payload)::

    {"event_id": "...", "source": "synthetic-fhir-r4", "sequence": 42,
     "received_at": "RFC3339", "resource": {"resourceType": "Observation", ...}}

MockFhirSource  — implemented: replays a versioned synthetic fixture.
EpicFhirSource  — contract and configuration boundary only (Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Protocol

from ..config import FIXTURE_DIR
SYNTHETIC_SOURCE_SYSTEM = "synthetic-fhir-r4"


@dataclass(frozen=True)
class SourceEvent:
    source_sequence: int          # stable per-source ordinal, 1-based
    arrival_offset_seconds: float  # simulated arrival time at 1x, not clinical time
    resource: dict[str, Any]
    scenario: str = "routine"      # fixture annotation only; never used for logic


# How a source is played back, declared by the source rather than assumed by the
# page. A speed is a simulated arrival clock over a fixture; it is never a claim
# about how fast real data arrives, which is why the labels say "demo" and the
# stress test is declared apart from the two speeds a reader should see.
# A real hospital feed declares kind="realtime" and no speeds at all: data
# appears when it arrives, so there is nothing to select.
DEMO_SPEEDS = [
    {"value": 1.0, "label": "Normal demo"},
    {"value": 5.0, "label": "Fast demo"},
]
STRESS_SPEEDS = [
    {"value": 20.0, "label": "Stress test",
     "note": "Stress test only. This is not the speed of real hospital data."},
]


class ClinicalDataSource(Protocol):
    source_system: str
    fixture_version: str

    def initial_bundle(self) -> dict[str, Any]: ...

    def events(self) -> list[SourceEvent]: ...


class MockFhirSource:
    """Synthetic FHIR R4 feed loaded from a committed, versioned fixture."""

    source_system = SYNTHETIC_SOURCE_SYSTEM

    def __init__(self, path: Path | None = None):
        self.path = path or (FIXTURE_DIR / "synthetic_patient_v1.json")
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema") != "oslian.synthetic-fhir-stream.v1":
            raise ValueError("unsupported synthetic stream fixture schema")
        if document.get("synthetic") is not True:
            raise ValueError("fixture must declare synthetic: true")
        self.fixture_version = str(document["fixture_version"])
        self.patient_id = str(document["patient_id"])
        self.encounter_id = str(document["encounter_id"])
        self._bundle = document["initial_bundle"]
        self._events = tuple(
            SourceEvent(int(e["source_sequence"]), float(e["arrival_offset_seconds"]), e["resource"],
                        str(e.get("scenario", "routine")))
            for e in document["events"]
        )
        sequences = [e.source_sequence for e in self._events]
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            raise ValueError("fixture events must have unique increasing source_sequence")

    def initial_bundle(self) -> dict[str, Any]:
        return self._bundle

    def events(self) -> list[SourceEvent]:
        return list(self._events)

    def describe(self) -> dict[str, Any]:
        return {
            "source_key": "synthetic", "mode": "live_synthetic", "source_system": self.source_system,
            "label": "Live synthetic stream", "fixture_version": self.fixture_version,
            "time_axis": "calendar", "synthetic": True, "retrospective": False,
            "display_name": "Synthetic patient SYN-0001",
            "provenance": "Synthetic FHIR R4 fixture; no real patient data; not a hospital connection.",
            "step_label": "Next event",
            "playback": {
                "kind": "simulated",
                "step": True,
                "speeds": DEMO_SPEEDS,
                "developer_speeds": STRESS_SPEEDS,
                "note": "A simulated arrival clock over a fixture. Clinical timestamps inside the "
                        "resources are never changed by it.",
            },
        }


RETROSPECTIVE_EPOCH = "2014-01-01T00:00:00Z"   # replay convention: case time is hours since ICU admission
RETROSPECTIVE_SOURCE_SYSTEM = "retrospective-eicu-crd-demo-2.0.1"
# (variable -> LOINC code, UCUM unit) for the public case's canonical columns; every pair is a
# mapping.py rule, so the converted resources go through the same registry as the synthetic feed.
_RETRO_CODES = {
    "bun": ("3094-0", "mg/dL"), "bicarbonate": ("1963-8", "mmol/L"), "anion_gap": ("33037-3", "mmol/L"),
    "creatinine": ("2160-0", "mg/dL"), "glucose": ("2345-7", "mg/dL"), "potassium": ("2823-3", "mmol/L"),
    "sodium": ("2951-2", "mmol/L"), "heart_rate": ("8867-4", "/min"), "respiratory_rate": ("9279-1", "/min"),
    "o2sat": ("59408-5", "%"), "map": ("8478-0", "mm[Hg]"), "temperature": ("8310-5", "Cel"),
}
_RETRO_VITALS = {"heart_rate", "respiratory_rate", "o2sat", "map", "temperature", "urine_output"}


class RetrospectiveCaseSource:
    """Retrospective research replay of the deidentified public eICU demo stay.

    The old replay rows (``hours_since_onset`` + value + ``<variable>_age_hr``)
    are converted into FHIR R4 Observations at the *measurement* time
    (``hours - age``) with ``issued`` at the row time, so the same ledger,
    mapping registry, timeline, runtime and chart serve both modes. Nothing
    after the revealed row exists in the session until it is delivered.
    """

    source_system = RETROSPECTIVE_SOURCE_SYSTEM

    def __init__(self, path: Path | None = None):
        self.path = path or (FIXTURE_DIR / "retrospective_eicu_demo_case.json")
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "oslian.retrospective-cases.v1" or document.get("retrospective") is not True:
            raise ValueError("unsupported retrospective case fixture")
        self.case = document["cases"][0]
        self.fixture_version = f"retrospective:{self.case['id']}"
        self.patient_id = str(self.case["id"])
        self.encounter_id = f"{self.case['id']}-icu-stay"
        self._epoch = datetime.fromisoformat(RETROSPECTIVE_EPOCH.replace("Z", "+00:00"))
        self._events = self._convert_events()

    def _ts(self, hours: float) -> str:
        return (self._epoch + timedelta(hours=float(hours))).isoformat().replace("+00:00", "Z")

    def _convert_events(self) -> tuple[SourceEvent, ...]:
        events: list[SourceEvent] = []
        sequence = 0
        for index, event in enumerate(self.case["observation_events"], start=1):
            row = event["observation"]
            hours = float(row["hours_since_onset"])
            arrival = 4.0 * index
            for variable, (code, unit) in _RETRO_CODES.items():
                if row.get(variable) is None:
                    continue
                age = float(row.get(f"{variable}_age_hr") or 0.0)
                sequence += 1
                events.append(SourceEvent(sequence, arrival, self._observation(
                    variable, code, unit, float(row[variable]), hours - age, hours), f"row_{index}"))
            if row.get("urine_output") is not None:
                age = float(row.get("urine_output_age_hr") or 0.0)
                effective = hours - age
                sequence += 1
                resource = self._observation("urine_output", "9187-6", "mL", float(row["urine_output"]), effective, hours)
                resource.pop("effectiveDateTime")
                resource["effectivePeriod"] = {"start": self._ts(effective - 1.0), "end": self._ts(effective)}
                events.append(SourceEvent(sequence, arrival, resource, f"row_{index}"))
        return tuple(events)

    def _observation(self, variable: str, code: str, unit: str, value: float, effective: float, issued: float) -> dict:
        stamp = f"{effective:.4f}".replace(".", "p").replace("-", "m")
        return {
            "resourceType": "Observation",
            "id": f"{self.case['id']}-{variable}-h{stamp}",
            "meta": {"versionId": "1", "lastUpdated": self._ts(issued)},
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                      "code": "vital-signs" if variable in _RETRO_VITALS else "laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "encounter": {"reference": f"Encounter/{self.encounter_id}"},
            "effectiveDateTime": self._ts(effective),
            "issued": self._ts(issued),
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit},
        }

    def initial_bundle(self) -> dict[str, Any]:
        context = self.case.get("case_context", {})
        patient = {
            "resourceType": "Patient", "id": self.patient_id,
            "meta": {"versionId": "1", "lastUpdated": self._ts(0)},
            "identifier": [{"system": "https://physionet.org/content/eicu-crd-demo/2.0.1/patientunitstayid",
                            "value": str(self.case["source_metadata"]["patient_unit_stay_id"])}],
            "name": [{"text": self.case.get("display_name", "Demo ICU patient")}],
            "gender": str(context.get("gender", "unknown")).lower() if context.get("gender") else "unknown",
        }
        encounter = {
            "resourceType": "Encounter", "id": self.encounter_id,
            "meta": {"versionId": "1", "lastUpdated": self._ts(0)},
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP"},
            "subject": {"reference": f"Patient/{self.patient_id}"},
            "period": {"start": self._ts(0)},
        }
        return {"resourceType": "Bundle", "id": f"{self.case['id']}-bundle", "type": "searchset", "total": 2,
                "entry": [{"resource": patient, "search": {"mode": "match"}},
                          {"resource": encounter, "search": {"mode": "include"}}]}

    def events(self) -> list[SourceEvent]:
        return list(self._events)

    def describe(self) -> dict[str, Any]:
        meta = self.case.get("source_metadata", {})
        context = self.case.get("case_context", {})
        return {
            "source_key": "retrospective", "mode": "retrospective", "source_system": self.source_system,
            "label": "Retrospective replay", "fixture_version": self.fixture_version,
            "time_axis": "hours_since_admission", "epoch": RETROSPECTIVE_EPOCH, "synthetic": False,
            "retrospective": True, "display_name": self.case.get("display_name"),
            "title": self.case.get("title"), "case_source": self.case.get("case_source"),
            "provenance": f"{meta.get('dataset')} {meta.get('version')} · {meta.get('deidentification')} · {meta.get('license')}",
            "context": {k: context.get(k) for k in ("age", "gender", "unit_type", "source_admission_label")},
            "rows": [{"row": i + 1, "hours": float(e["observation"]["hours_since_onset"]),
                      "variables": sorted(k for k in e["observation"] if not k.endswith("_age_hr") and k != "hours_since_onset")}
                     for i, e in enumerate(self.case["observation_events"])],
            "step_label": "Next observation",
            "playback": {
                "kind": "simulated",
                "step": True,
                "speeds": DEMO_SPEEDS,
                # a three-row case has nothing to stress; the backend still accepts 20x
                "developer_speeds": [],
                "note": "The recorded case is revealed one observation at a time, or on a "
                        "simulated clock. Case time is hours since ICU admission.",
            },
        }


SOURCES: dict[str, type] = {"synthetic": MockFhirSource, "retrospective": RetrospectiveCaseSource}


def make_source(key: str = "synthetic"):
    if key not in SOURCES:
        raise KeyError(key)
    return SOURCES[key]()


@dataclass(frozen=True)
class EpicSourceConfig:
    """Server-side configuration for a future Epic connection.

    Secrets are referenced by environment-variable *name* only; nothing here
    holds a token, key, or client secret, and nothing is committed.
    """

    fhir_base_url: str                       # e.g. https://<org>/api/FHIR/R4 (no trailing slash)
    client_id: str                           # Epic-registered backend client id (not a secret)
    private_key_env: str = "OSLIAN_EPIC_PRIVATE_KEY_PEM"   # RS384 key for the JWT client assertion
    scopes: tuple[str, ...] = ("system/Patient.read", "system/Encounter.read", "system/Observation.read")
    scope_dialect: str = "smart-v1"          # or "smart-v2" (system/Observation.rs), per the organization grant
    launch: str = "backend-services"         # or "ehr-launch" for a clinician-facing embedded app
    organization: str = ""
    page_size: int = 200
    extra: dict[str, Any] = field(default_factory=dict)


class EpicFhirSource:
    """Contract boundary for Epic on FHIR. Not connected in this phase.

    Design (from the SMART App Launch 2.0.0 Backend Services profile and Epic's
    published documentation), fixed here so the rest of the pipeline never has
    to change when it is implemented:

    * discovery: ``GET {fhir_base_url}/.well-known/smart-configuration`` gives
      ``token_endpoint`` and supported scopes/capabilities;
    * unattended reads use SMART Backend Services: a one-time-use JWT client
      assertion (``client_assertion_type`` =
      ``urn:ietf:params:oauth:client-assertion-type:jwt-bearer``,
      ``grant_type=client_credentials``) signed with a server-held private key;
      ``system/`` scopes are requested in the dialect the organization granted;
    * a clinician-facing app launched from within Epic uses SMART EHR launch
      instead (``launch`` + ``iss`` parameters, authorization-code flow);
    * incremental reads are triggered by a hospital-side event (ADT, results
      interface, or a FHIR Subscription only when the organization's
      CapabilityStatement declares support) and then fetch deltas with
      ``GET Observation?patient=...&category=...&_lastUpdated=gt<watermark>``,
      following ``Bundle.link[rel=next]`` until exhausted; there is no
      continuous polling and Bulk FHIR is not treated as a stream;
    * every fetched resource is handed downstream as the same
      ``SourceEvent`` envelope the mock source produces.
    """

    source_system = "epic-fhir-r4"
    fixture_version = "n/a"

    def __init__(self, config: EpicSourceConfig):
        self.config = config

    def describe(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "status": "not_connected",
            "phase": "P3",
            # A real feed has no playback: resources appear when the hospital
            # sends them. Declared here so the page hides the speed control from
            # source metadata rather than from a hard-coded exception.
            "playback": {"kind": "realtime", "step": False, "speeds": [], "developer_speeds": [],
                         "note": "Live data appears at its actual arrival time; there is no speed to choose."},
            "fhir_base_url": self.config.fhir_base_url,
            "auth": {
                "backend": "SMART Backend Services (client_credentials + JWT client assertion)",
                "clinician_app": "SMART EHR launch (authorization code)",
                "discovery": "/.well-known/smart-configuration",
                "secrets": "server-side only; referenced by environment variable name",
                "scope_dialect": self.config.scope_dialect,
                "scopes": list(self.config.scopes),
            },
            "read_model": {
                "trigger": "hospital-side event -> incremental FHIR search with _lastUpdated watermark",
                "pagination": "Bundle.link rel=next",
                "subscription": "only if the organization CapabilityStatement supports it",
                "polling": False,
                "bulk_fhir_as_stream": False,
            },
        }

    def initial_bundle(self) -> dict[str, Any]:
        raise NotImplementedError("EpicFhirSource is a Phase 3 deliverable; no sandbox credentials in this build")

    def events(self) -> list[SourceEvent]:
        raise NotImplementedError("EpicFhirSource is a Phase 3 deliverable; no sandbox credentials in this build")

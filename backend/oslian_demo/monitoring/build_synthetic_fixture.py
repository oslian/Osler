"""Generate the versioned synthetic FHIR R4 stream fixture.

Everything here is synthetic: ids are prefixed ``syn-``, the patient has no
demographics beyond ``gender: unknown``, and the observed values follow a
gentle seeded drift with no diagnostic narrative. Arrival offsets are a demo
delivery schedule at 1x and are unrelated to the clinical timestamps.

    python -m oslian_demo.monitoring.build_synthetic_fixture --write
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import random

from ..config import FIXTURE_DIR

FIXTURE_PATH = FIXTURE_DIR / "synthetic_patient_v1.json"
FIXTURE_VERSION = "synthetic_patient_v1"
SOURCE_SYSTEM = "synthetic-fhir-r4"
PATIENT_ID = "syn-pat-0001"
ENCOUNTER_ID = "syn-enc-0001"
ONSET = datetime(2026, 3, 14, 6, 0, tzinfo=timezone.utc)
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
DAR = "http://terminology.hl7.org/CodeSystem/data-absent-reason"

VITALS = {
    # variable: (code, display, unit, start, end, noise, decimals)
    "heart_rate": ("8867-4", "Heart rate", "/min", 96, 84, 3.0, 0),
    "o2sat": ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%", 93, 96, 0.8, 0),
    "respiratory_rate": ("9279-1", "Respiratory rate", "/min", 24, 18, 1.5, 0),
}
LABS = {
    "glucose": ("2345-7", "Glucose [Mass/volume] in Serum or Plasma", "mg/dL", 212, 150, 8.0, 0),
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL", 1.5, 1.2, 0.05, 2),
    "potassium": ("2823-3", "Potassium [Moles/volume] in Serum or Plasma", "mmol/L", 4.6, 4.1, 0.12, 1),
    "sodium": ("2951-2", "Sodium [Moles/volume] in Serum or Plasma", "mmol/L", 137, 139, 0.8, 0),
    "bicarbonate": ("1963-8", "Bicarbonate [Moles/volume] in Serum or Plasma", "mmol/L", 19, 23, 0.6, 0),
    "chloride": ("2075-0", "Chloride [Moles/volume] in Serum or Plasma", "mmol/L", 104, 105, 0.9, 0),
    "bun": ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma", "mg/dL", 32, 26, 1.2, 0),
    "hemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL", 10.4, 10.1, 0.15, 1),
    "hematocrit": ("4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count", "%", 31.2, 30.5, 0.4, 1),
    "platelets": ("777-3", "Platelets [#/volume] in Blood by Automated count", "10*3/uL", 172, 168, 5.0, 0),
    "wbc": ("6690-2", "Leukocytes [#/volume] in Blood by Automated count", "10*3/uL", 14.2, 11.0, 0.5, 1),
    "lactate": ("2524-7", "Lactate [Moles/volume] in Serum or Plasma", "mmol/L", 2.8, 1.3, 0.15, 1),
}


def _ts(hours: float) -> str:
    return (ONSET + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def _drift(start: float, end: float, noise: float, hours: float, rng: random.Random, decimals: int,
           span: float = 30.0) -> float:
    frac = min(max(hours / span, 0.0), 1.0)
    value = start + (end - start) * frac + rng.gauss(0.0, noise)
    return round(value, decimals) if decimals else float(round(value))


def _observation(obs_id: str, version: str, status: str, category: str, code: str, display: str,
                 effective_hours: float, issued_minutes: float, **extra) -> dict:
    resource = {
        "resourceType": "Observation",
        "id": obs_id,
        "meta": {"versionId": version, "lastUpdated": _ts(effective_hours + issued_minutes / 60.0)},
        "status": status,
        "category": [{"coding": [{"system": CATEGORY, "code": category,
                                  "display": "Vital Signs" if category == "vital-signs" else "Laboratory"}]}],
        "code": {"coding": [{"system": LOINC, "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "encounter": {"reference": f"Encounter/{ENCOUNTER_ID}"},
    }
    if "period_hours" in extra:
        resource["effectivePeriod"] = {"start": _ts(effective_hours - extra.pop("period_hours")),
                                       "end": _ts(effective_hours)}
    else:
        resource["effectiveDateTime"] = _ts(effective_hours)
    resource["issued"] = _ts(effective_hours + issued_minutes / 60.0)
    resource.update(extra)
    return resource


def _quantity(value: float, unit: str, unit_code: str | None = None) -> dict:
    quantity = {"value": value, "unit": unit}
    if unit_code is not None:
        quantity["system"] = UCUM
        quantity["code"] = unit_code
    return quantity


def build(seed: int = 20260314) -> dict:
    rng = random.Random(seed)
    clinical: list[tuple[float, dict, str]] = []   # (effective_hours, resource, scenario)

    def add(hours: float, resource: dict, scenario: str = "routine") -> None:
        clinical.append((hours, resource, scenario))

    # vitals every 2 h for 30 h; MAP alternates between a BP panel and a direct MAP code
    for step in range(0, 31, 2):
        h = float(step)
        for variable, (code, display, unit, start, end, noise, dec) in VITALS.items():
            value = _drift(start, end, noise, h, rng, dec)
            add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "vital-signs", code, display, h, 1,
                                valueQuantity=_quantity(value, unit, unit)))
        map_value = _drift(68, 78, 2.0, h, rng, 0)
        if step % 4 == 0:
            systolic = float(round(map_value + 34 + rng.gauss(0, 2)))
            diastolic = float(round((3 * map_value - systolic) / 2))
            add(h, _observation(f"syn-obs-bp-panel-h{step:02d}", "1", "final", "vital-signs", "85354-9",
                                "Blood pressure panel with all children optional", h, 1,
                                component=[
                                    {"code": {"coding": [{"system": LOINC, "code": "8480-6", "display": "Systolic blood pressure"}]},
                                     "valueQuantity": _quantity(systolic, "mmHg", "mm[Hg]")},
                                    {"code": {"coding": [{"system": LOINC, "code": "8462-4", "display": "Diastolic blood pressure"}]},
                                     "valueQuantity": _quantity(diastolic, "mmHg", "mm[Hg]")},
                                ]), "bp_panel_components")
        else:
            add(h, _observation(f"syn-obs-map-h{step:02d}", "1", "final", "vital-signs", "8478-0",
                                "Mean blood pressure", h, 1, valueQuantity=_quantity(map_value, "mmHg", "mm[Hg]")))
        if step % 4 == 0:
            temp_c = _drift(38.2, 37.3, 0.15, h, rng, 1)
            if step == 8:
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)
                add(h, _observation(f"syn-obs-temperature-h{step:02d}", "1", "final", "vital-signs", "8310-5",
                                    "Body temperature", h, 1, valueQuantity=_quantity(temp_f, "degF", "[degF]")),
                    "fahrenheit_unit_conversion")
            else:
                add(h, _observation(f"syn-obs-temperature-h{step:02d}", "1", "final", "vital-signs", "8310-5",
                                    "Body temperature", h, 1, valueQuantity=_quantity(temp_c, "Cel", "Cel")))
    # urine output: 2-hour collection periods reported as a volume over effectivePeriod
    for step in range(2, 31, 2):
        h = float(step)
        rate = _drift(45, 75, 6.0, h, rng, 0)
        add(h, _observation(f"syn-obs-urine-h{step:02d}", "1", "final", "vital-signs", "9187-6", "Urine output",
                            h, 5, period_hours=2.0, valueQuantity=_quantity(rate * 2.0, "mL", "mL")),
            "volume_over_period")
    # labs every 6 h
    for step in range(0, 31, 6):
        h = float(step)
        for variable, (code, display, unit, start, end, noise, dec) in LABS.items():
            value = _drift(start, end, noise, h, rng, dec)
            if variable == "glucose" and step == 18:
                add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", "14749-6",
                                    "Glucose [Moles/volume] in Serum or Plasma", h, 22,
                                    valueQuantity=_quantity(round(value / 18.0156, 1), "mmol/L", "mmol/L")),
                    "mmol_unit_conversion")
                continue
            if variable == "glucose" and step == 24:
                add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", code, display, h,
                                    22, valueQuantity=_quantity(value, "mg/dL")), "missing_ucum_unit")
                continue
            if variable == "creatinine" and step == 30:
                add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", code, display, h,
                                    22, dataAbsentReason={"coding": [{"system": DAR, "code": "not-performed",
                                                                      "display": "Not Performed"}]}),
                    "data_absent_reason")
                continue
            if variable == "potassium" and step == 12:
                add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", code, display, h,
                                    22, valueQuantity=_quantity(5.9, "mmol/L", "mmol/L")), "amended_v1")
                continue
            if variable == "lactate" and step == 18:
                add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", code, display, h,
                                    35, valueQuantity=_quantity(value, "mmol/L", "mmol/L")), "out_of_order")
                continue
            add(h, _observation(f"syn-obs-{variable}-h{step:02d}", "1", "final", "laboratory", code, display, h,
                                22, valueQuantity=_quantity(value, unit, unit)))
    # unmapped code: body height charted once
    add(2.0, _observation("syn-obs-height-h02", "1", "final", "vital-signs", "8302-2", "Body height", 2.0, 1,
                          valueQuantity=_quantity(171.0, "cm", "cm")), "unmapped_code")
    # entered-in-error correction of a temperature at 26 h (v1 wrong, v2 retracts)
    add(26.0, _observation("syn-obs-temperature-h26", "1", "final", "vital-signs", "8310-5", "Body temperature",
                           26.0, 1, valueQuantity=_quantity(39.9, "Cel", "Cel")), "entered_in_error_v1")

    clinical.sort(key=lambda item: (item[0], item[1]["id"]))

    # snapshot = everything at onset (hour 0)
    snapshot = [r for h, r, _ in clinical if h == 0.0]
    stream = [(h, r, s) for h, r, s in clinical if h > 0.0]

    events = []
    offset = 4.0
    for hours, resource, scenario in stream:
        offset += round(rng.uniform(2.5, 5.5), 1)
        events.append({"arrival_offset_seconds": round(offset, 1), "scenario": scenario, "resource": resource})

    def insert_after(resource_id_predicate, new_event, delay):
        for index, event in enumerate(events):
            if resource_id_predicate(event["resource"]["id"]):
                new_event["arrival_offset_seconds"] = round(events[index]["arrival_offset_seconds"] + delay, 1)
                events.insert(index + 1, new_event)
                return
        raise KeyError("anchor event not found")

    # duplicate delivery of the 10 h heart-rate resource (identical id + versionId)
    hr10 = next(r for h, r, _ in clinical if r["id"] == "syn-obs-heart_rate-h10")
    insert_after(lambda rid: rid == "syn-obs-o2sat-h10", {"scenario": "duplicate_delivery", "resource": hr10}, 1.5)
    # amended potassium at 12 h: version 2 corrects the value
    k12 = next(r for h, r, _ in clinical if r["id"] == "syn-obs-potassium-h12")
    amended = json.loads(json.dumps(k12))
    amended["meta"] = {"versionId": "2", "lastUpdated": _ts(12.0 + 95 / 60.0)}
    amended["status"] = "amended"
    amended["issued"] = _ts(12.0 + 95 / 60.0)
    amended["valueQuantity"] = _quantity(4.4, "mmol/L", "mmol/L")
    insert_after(lambda rid: rid == "syn-obs-heart_rate-h14", {"scenario": "amended_v2", "resource": amended}, 2.0)
    # out-of-order: the 18 h lactate arrives after the 20 h vitals
    lactate_index = next(i for i, e in enumerate(events) if e["resource"]["id"] == "syn-obs-lactate-h18")
    lactate_event = events.pop(lactate_index)
    insert_after(lambda rid: rid == "syn-obs-respiratory_rate-h20", lactate_event, 1.8)
    # entered-in-error: version 2 of the 26 h temperature retracts version 1
    t26 = next(r for h, r, _ in clinical if r["id"] == "syn-obs-temperature-h26")
    retracted = json.loads(json.dumps(t26))
    retracted["meta"] = {"versionId": "2", "lastUpdated": _ts(26.0 + 20 / 60.0)}
    retracted["status"] = "entered-in-error"
    retracted.pop("valueQuantity")
    insert_after(lambda rid: rid == "syn-obs-heart_rate-h28", {"scenario": "entered_in_error_v2", "resource": retracted}, 1.2)

    # re-sequence arrival offsets so they stay strictly increasing after insertions
    last = 0.0
    for index, event in enumerate(events, start=1):
        if event["arrival_offset_seconds"] <= last:
            event["arrival_offset_seconds"] = round(last + 1.0, 1)
        last = event["arrival_offset_seconds"]
        event["source_sequence"] = index

    patient = {
        "resourceType": "Patient", "id": PATIENT_ID,
        "meta": {"versionId": "1", "lastUpdated": _ts(0)},
        "identifier": [{"system": "urn:oslian:synthetic-patient", "value": "SYN-0001"}],
        "active": True, "name": [{"text": "Synthetic patient SYN-0001"}], "gender": "unknown",
    }
    encounter = {
        "resourceType": "Encounter", "id": ENCOUNTER_ID,
        "meta": {"versionId": "1", "lastUpdated": _ts(0)},
        "status": "in-progress",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP", "display": "inpatient encounter"},
        "subject": {"reference": f"Patient/{PATIENT_ID}"},
        "period": {"start": _ts(0)},
    }
    bundle = {
        "resourceType": "Bundle", "id": "syn-bundle-0001", "type": "searchset",
        "meta": {"lastUpdated": _ts(0)},
        "total": 2 + len(snapshot),
        "entry": [{"fullUrl": f"urn:oslian:synthetic/Patient/{PATIENT_ID}", "resource": patient, "search": {"mode": "match"}},
                  {"fullUrl": f"urn:oslian:synthetic/Encounter/{ENCOUNTER_ID}", "resource": encounter, "search": {"mode": "include"}}]
                 + [{"fullUrl": f"urn:oslian:synthetic/Observation/{r['id']}", "resource": r, "search": {"mode": "include"}}
                    for r in snapshot],
    }
    return {
        "schema": "oslian.synthetic-fhir-stream.v1",
        "fixture_version": FIXTURE_VERSION,
        "synthetic": True,
        "phi": False,
        "generated_by": "oslian_demo/monitoring/build_synthetic_fixture.py",
        "seed": seed,
        "source_system": SOURCE_SYSTEM,
        "patient_id": PATIENT_ID,
        "encounter_id": ENCOUNTER_ID,
        "notes": [
            "All identifiers are synthetic (syn-*). No real patient data was used.",
            "Clinical timestamps are fixed inside each resource; arrival_offset_seconds is the demo delivery schedule at 1x.",
            "Scenarios: duplicate_delivery, amended_v2, out_of_order, missing_ucum_unit, data_absent_reason, unmapped_code, entered_in_error_v2, fahrenheit_unit_conversion, mmol_unit_conversion, bp_panel_components, volume_over_period.",
        ],
        "initial_bundle": bundle,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    document = build()
    text = json.dumps(document, indent=1, ensure_ascii=False) + "\n"
    if args.write:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {FIXTURE_PATH} ({len(document['events'])} events, {len(document['initial_bundle']['entry'])} bundle entries)")
    else:
        print(json.dumps({"events": len(document["events"]), "bundle_entries": len(document["initial_bundle"]["entry"])}))


if __name__ == "__main__":
    main()

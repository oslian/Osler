"""Build one replay case per ward stay, so a row on the board can be opened.

The ward answers "who first". Opening a row should then show that stay the way
the rest of the page shows a stay -- the timeline, the organ figure, the FHIR
provenance -- and that path needs a replay case per stay. Until now there was
exactly one case, and it was not on the ward.

Every value here is a real measurement from the deidentified public eICU CRD
Demo 2.0.1: the observations come from the same anchor trajectories the board
reads, and the demographics from the dataset's own `patient` table. Nothing is
generated, interpolated or carried forward.

Two things this deliberately does not write:

*No ``<variable>_age_hr``.* In the older hand-built case a row could carry a
value measured some hours before the row's own timestamp, and the age said how
far. An anchor trajectory row *is* the measurement time, so the age is zero for
every value here -- and writing a field that is always zero would suggest a
measurement age was recorded when none was.

*No variable without a mapping rule.* A value the mapping registry cannot code
would arrive as an unmapped observation, so the selection below is the
intersection of what the stays actually have and what ``mapping.py`` already
codes. The registry is not extended to fit the data.

    python -m oslian_demo.monitoring.build_retrospective_cases \
        --anchors <path>/anchors_eicu_tx.jsonl \
        --patient-table <path>/eicu-crd-demo/2.0.1/patient.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

from ..config import FIXTURE_DIR

SCHEMA = "oslian.retrospective-cases.v1"
DEFAULT_WARD = FIXTURE_DIR / "ward_eicu_demo_v1.json"
DEFAULT_OUT = FIXTURE_DIR / "retrospective_eicu_ward_cases.json"

# The laboratory record, plus the two bedside signals recorded sparsely enough to
# ride with it. Heart rate, oxygen saturation and respiratory rate are left out
# on purpose: they are 36,000 of the 44,000 values these stays hold, charted
# every few minutes, and carrying them would make a stay take an hour to replay
# without adding anything the board or the renal view reads. The document says so
# in `variable_scope` rather than leaving the gap to be discovered.
CASE_VARIABLES = (
    "creatinine", "bun", "potassium", "sodium", "chloride", "bicarbonate",
    "anion_gap", "glucose", "calcium", "magnesium", "phosphate", "albumin",
    "lactate", "ph", "hemoglobin", "platelets", "wbc",
    "urine_output", "map", "temperature",
)

VARIABLE_SCOPE = (
    "Laboratory results, urine output, mean arterial pressure and temperature. "
    "Heart rate, oxygen saturation and respiratory rate are recorded every few "
    "minutes in this dataset and are omitted so a stay stays replayable; they "
    "are absent from these cases, not missing from the source."
)

SOURCE_METADATA = {
    "dataset": "eICU Collaborative Research Database Demo",
    "version": "2.0.1",
    "url": "https://physionet.org/content/eicu-crd-demo/2.0.1/",
    "license": "Open Data Commons Open Database License v1.0",
    "deidentification": "PhysioNet demo data; HIPAA safe-harbor deidentified",
}


def _text(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def load_context(path: Path | None) -> dict[str, dict[str, Any]]:
    """`patientunitstayid` -> the demographic columns the dataset itself records.

    eICU writes an age of ">89" rather than a number for the oldest patients, so
    it is carried as the string the dataset wrote and never coerced.
    """
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["patientunitstayid"]] = {
                "age": _text(row.get("age")),
                "gender": _text(row.get("gender")),
                "unit_type": _text(row.get("unittype")),
                "source_admission_label": _text(row.get("apacheadmissiondx")),
            }
    return out


def observation_events(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One event per charted moment, carrying only codeable variables."""
    events = []
    for row in trajectory:
        observation = {"hours_since_onset": round(float(row["hours_since_onset"]), 4)}
        for variable in CASE_VARIABLES:
            if row.get(variable) is not None:
                observation[variable] = float(row[variable])
        if len(observation) > 1:
            events.append({"observation": observation})
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--ward", type=Path, default=DEFAULT_WARD,
                        help="the board these cases must cover, id for id")
    parser.add_argument("--patient-table", type=Path, default=None,
                        help="eICU patient.csv.gz, for the recorded demographics")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ward = json.loads(args.ward.read_text(encoding="utf-8"))
    # Driven by the board rather than by re-running its selection: every row on
    # the board is then guaranteed a case, and the two cannot drift apart.
    wanted = {p["id"]: p for p in ward["ward"]["patients"]}
    encounter_of = {p["id"].replace("eicu-demo-", ""): p["id"] for p in wanted.values()}
    context = load_context(args.patient_table)

    latest: dict[str, dict[str, Any]] = {}
    with args.anchors.open() as handle:
        for line in handle:
            anchor = json.loads(line)
            encounter = str(anchor["encounter_id"])
            if encounter not in encounter_of:
                continue
            current = latest.get(encounter)
            if current is None or anchor["anchor_hour"] > current["anchor_hour"]:
                latest[encounter] = anchor

    cases = []
    for encounter, anchor in sorted(latest.items()):
        case_id = encounter_of[encounter]
        events = observation_events(anchor["trajectory"])
        if not events:
            continue
        cases.append({
            "id": case_id,
            "display_name": wanted[case_id]["display_name"],
            "title": f"{wanted[case_id]['display_name']} · {wanted[case_id].get('care_unit') or 'unknown unit'}",
            "case_source": "eICU CRD Demo 2.0.1 stay, replayed from its own recorded observations",
            "care_unit": wanted[case_id].get("care_unit"),
            "source_metadata": {**SOURCE_METADATA, "patient_unit_stay_id": int(encounter)},
            "case_context": context.get(encounter, {}),
            "observation_events": events,
        })

    missing = sorted(set(wanted) - {c["id"] for c in cases})
    document = {
        "schema_version": SCHEMA,
        "retrospective": True,
        "not_live": True,
        "note": ("One replay case per stay on the ward board, built from the same "
                 "deidentified public eICU CRD Demo 2.0.1 observations the board "
                 "reads. Every value is a recorded measurement; nothing is "
                 "interpolated, carried forward or generated."),
        "variable_scope": VARIABLE_SCOPE,
        "source_metadata": SOURCE_METADATA,
        "cases": cases,
    }
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    rows = sum(len(c["observation_events"]) for c in cases)
    values = sum(len(e["observation"]) - 1 for c in cases for e in c["observation_events"])
    print(f"wrote {args.out}: {len(cases)} cases, {rows} charted moments, {values} values")
    print(f"  {rows / max(1, len(cases)):.0f} moments and {values / max(1, len(cases)):.0f} "
          f"values per stay")
    if missing:
        # Every board row has to be openable; a gap here is a broken link.
        print(f"  WARNING: {len(missing)} ward stays have no case: {', '.join(missing)}")
    else:
        print(f"  every one of the {len(wanted)} ward stays has a case")


if __name__ == "__main__":
    main()

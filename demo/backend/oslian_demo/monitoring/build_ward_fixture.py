"""Build the ward fixture from the deidentified eICU CRD Demo anchor set.

The replay case fixture is one stay. An attention order needs a ward, so this
converts N stays from the same public dataset into one board-shaped document:
each patient's observed creatinine and urine-output history up to a shared
wall clock, plus the treatment-window flags the anchor builder recorded.

Input is the head-to-head anchor file (`anchors_eicu_tx.jsonl`), which already
carries a causal trajectory prefix per anchor and never contains anything after
its own anchor hour. Nothing here reads a label or a forecast.

Body weight is the one field the anchors do not carry, and without it a charted
urine rate in mL/hour cannot be compared to anything. It is joined here from the
dataset's own `patient` table on `patientunitstayid`, taking the admission
weight and falling back to the discharge weight. A stay with neither, or with a
weight outside the range `ranking.PLAUSIBLE_WEIGHT_KG` will divide by, is
written as `null` -- explicitly absent rather than quietly omitted, and never
filled in from the rest of the ward.

    python -m oslian_demo.monitoring.build_ward_fixture \
        --anchors <path>/anchors_eicu_tx.jsonl --patients 40 \
        --patient-table <path>/eicu-crd-demo/2.0.1/patient.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any

from ..config import FIXTURE_DIR
from .ranking import usable_weight

SCHEMA = "oslian.ward-fixture.v1"
DEFAULT_OUT = "ward_eicu_demo_v1.json"
# Creatinine is the score and urine output is reported per kg; the rest are here
# because the band a stay is listed under reads every analyte that has cutpoints,
# not just the one that orders the list.
SERIES_VARIABLES = ("creatinine", "urine_output",
                    "bun", "potassium", "sodium", "chloride", "bicarbonate")
TREATMENT_KEYS = ("hist_fluids", "hist_vasopressor", "hist_diuretics",
                  "hist_renal_replacement", "hist_nephrotoxin")

SOURCE_METADATA = {
    "dataset": "eICU Collaborative Research Database Demo",
    "version": "2.0.1",
    "url": "https://physionet.org/content/eicu-crd-demo/2.0.1/",
    "license": "Open Data Commons Open Database License v1.0",
    "deidentification": "PhysioNet demo data; HIPAA safe-harbor deidentified",
}

SAFETY_BOUNDARY = {
    "factual_research_only": True,
    "diagnostic_claim_allowed": False,
    "causal_claim_allowed": False,
    "treatment_recommendation_allowed": False,
    "automated_prescribing_allowed": False,
    "clinical_alert": False,
}


def series(trajectory: list[dict[str, Any]], variable: str) -> list[list[float]]:
    return [[round(float(row["hours_since_onset"]), 4), float(row[variable])]
            for row in trajectory if variable in row]


def _number(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_weights(path: Path | None) -> dict[str, float]:
    """`patientunitstayid` -> a weight in kg the board is willing to divide by.

    Admission weight first, discharge weight as the fallback -- the same two
    columns and the same order for every stay, so which one a row got does not
    depend on anything about that row. Anything the plausibility check rejects is
    left out entirely, which is the same outcome as a stay the table never had.
    """
    if path is None:
        return {}
    weights: dict[str, float] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            weight = (usable_weight(_number(row.get("admissionweight")))
                      or usable_weight(_number(row.get("dischargeweight"))))
            if weight is not None:
                weights[row["patientunitstayid"]] = weight
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--out", type=Path, default=FIXTURE_DIR / DEFAULT_OUT)
    parser.add_argument("--patient-table", type=Path, default=None,
                        help="eICU patient.csv.gz, for the admission weight column")
    args = parser.parse_args()

    weights = load_weights(args.patient_table)

    # One row per patient: their latest anchor, which is the only one that can be
    # "now" on a board. Keeping two would put a stale row next to a fresh one.
    latest: dict[str, dict[str, Any]] = {}
    with args.anchors.open() as handle:
        for line in handle:
            anchor = json.loads(line)
            current = latest.get(anchor["patient_id"])
            if current is None or anchor["anchor_hour"] > current["anchor_hour"]:
                latest[anchor["patient_id"]] = anchor

    chosen = sorted(latest.values(), key=lambda a: str(a["patient_id"]))
    random.Random(args.seed).shuffle(chosen)

    # Keep a couple of patients with no creatinine on the board on purpose: the
    # "not scored" path is part of the contract and has to be visible in the demo.
    scored = [a for a in chosen if any("creatinine" in r for r in a["trajectory"])]
    unscored = [a for a in chosen if a not in scored]
    keep_unscored = min(2, len(unscored), max(0, args.patients - 1))
    selected = scored[: args.patients - keep_unscored] + unscored[:keep_unscored]
    selected.sort(key=lambda a: str(a["patient_id"]))

    patients = []
    for index, anchor in enumerate(selected, start=1):
        patients.append({
            "id": f"eicu-demo-{anchor['encounter_id']}",
            "display_name": f"Bed {index:02d}",
            "care_unit": anchor.get("care_unit") or "unknown",
            "now_hours": round(float(anchor["anchor_hour"]), 4),
            # null, not a missing key: a stay with no usable weight has to be
            # visible as one, and the board withholds the per-kg rate for it.
            "weight_kg": weights.get(str(anchor["encounter_id"])),
            **{v: series(anchor["trajectory"], v) for v in SERIES_VARIABLES},
            "treatments": {k: int(bool(anchor.get("treatments", {}).get(k)))
                           for k in TREATMENT_KEYS},
        })

    document = {
        "schema_version": SCHEMA,
        "retrospective": True,
        "not_live": True,
        "note": ("Deidentified public eICU CRD Demo 2.0.1 stays, one row per patient, "
                 "each carrying only the observations recorded before its own anchor hour. "
                 "No label, forecast or outcome from the head-to-head set is included."),
        "source_metadata": SOURCE_METADATA,
        "ward": {
            "label": "Deidentified eICU CRD Demo 2.0.1 stays",
            "variable_units": {"creatinine": "mg/dL", "urine_output": "mL/h",
                               "bun": "mg/dL", "potassium": "mmol/L", "sodium": "mmol/L",
                               "chloride": "mmol/L", "bicarbonate": "mmol/L",
                               "weight_kg": "kg (admission, else discharge; null when neither is usable)",
                               "now_hours": "h", "treatments": "active in the 6h before now_hours"},
            "patients": patients,
        },
        "safety_boundary": SAFETY_BOUNDARY,
    }
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    with_creatinine = sum(1 for p in patients if p["creatinine"])
    with_weight = sum(1 for p in patients if p["weight_kg"] is not None)
    with_rate = sum(1 for p in patients if p["weight_kg"] is not None and p["urine_output"])
    print(f"wrote {args.out}: {len(patients)} patients "
          f"({with_creatinine} with creatinine, {len(patients) - with_creatinine} without)")
    if args.patient_table is None:
        print("  no --patient-table: every stay is written without a weight, so the "
              "board states no rate per kg for any of them")
    print(f"  weight: {with_weight}/{len(patients)}; "
          f"urine rate per kg statable for {with_rate}")


if __name__ == "__main__":
    main()

"""Memory-bounded whole-ICU MIMIC-IV observation transition extractor.

This wrapper uses the existing observational extractor in stay batches so the
full MIMIC-IV tables do not have to be materialized in memory at once.  It is
intended for the 1h/3h waveform-linked cohorts and keeps the same ``*_t`` /
``*_tp{horizon}`` contract as the single-pass extractor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import mimiciv_observation_transition_extract as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=float, default=1.0)
    parser.add_argument("--anchor-step-hours", type=float, default=None)
    parser.add_argument("--target-tolerance-hours", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-stays", type=int, default=-1)
    parser.add_argument(
        "--stay-ids-file",
        type=Path,
        default=None,
        help="optional newline-delimited ICU stay_id filter, useful for waveform-linked cohorts",
    )
    parser.add_argument("--include-treatment-context", action="store_true")
    return parser.parse_args()


def configure(horizon: float, step: float | None, tolerance: float | None) -> str:
    if horizon <= 0:
        raise ValueError("--horizon-hours must be positive")
    if step is not None and step <= 0:
        raise ValueError("--anchor-step-hours must be positive")
    base.DELTA_H = float(horizon)
    base.ANCHOR_STEP_H = float(step or horizon)
    base.TARGET_TOL_H = float(tolerance if tolerance is not None else min(2.0, horizon / 2.0))
    if base.TARGET_TOL_H <= 0:
        raise ValueError("--target-tolerance-hours must be positive")
    return f"_tp{int(horizon)}" if float(horizon).is_integer() else f"_tp{horizon:g}"


def build_batch_frame(meta: pd.DataFrame, measurements: pd.DataFrame, future_suffix: str,
                      evidence: pd.DataFrame | None) -> pd.DataFrame:
    anchors = base.build_anchors(meta)
    state_t = base.assemble_state(anchors, measurements, "_t", "t", "backward", base.LOOKBACK_H)
    state_f = base.assemble_state(
        anchors, measurements, future_suffix, "t_plus", "nearest", base.TARGET_TOL_H
    )
    treatment_context = pd.DataFrame(index=anchors.index)
    if evidence is not None:
        treatment_context = base.assemble_treatment_context(anchors, evidence, base.DELTA_H)
    frame = pd.concat(
        [
            anchors[["subject_id", "stay_id", "first_careunit", "last_careunit", "t", "t_plus"]],
            state_t.drop(columns=["stay_id", "t"]),
            state_f.drop(columns=["stay_id", "t_plus"]),
            treatment_context,
        ],
        axis=1,
    )
    has_pair = np.zeros(len(frame), dtype=bool)
    for var in base.STATE_VARS:
        has_pair |= frame[f"{var}_t"].notna().to_numpy() & frame[f"{var}{future_suffix}"].notna().to_numpy()
    return frame.loc[has_pair].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    future_suffix = configure(args.horizon_hours, args.anchor_step_hours, args.target_tolerance_hours)
    max_stays = None if args.max_stays < 0 else int(args.max_stays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    connection = base.con()
    meta = base.pull_stay_meta(connection, args.mimic_dir, max_stays)
    if args.stay_ids_file is not None:
        stay_ids = pd.read_csv(args.stay_ids_file, header=None, names=["stay_id"])
        stay_ids["stay_id"] = pd.to_numeric(stay_ids["stay_id"], errors="coerce")
        stay_ids = set(stay_ids["stay_id"].dropna().astype("int64").tolist())
        meta = meta[meta["stay_id"].isin(stay_ids)].reset_index(drop=True)
        print(f"Waveform-linked stay filter: {len(stay_ids):,}; matched={len(meta):,}", flush=True)
    if meta.empty:
        raise ValueError("no ICU stays found")
    print(f"Selected ICU stays: {len(meta):,}; batch_size={args.batch_size}", flush=True)

    chunk_reports = []
    careunits = set()
    with tempfile.TemporaryDirectory(prefix="mimiciv_obs_chunks_", dir="/private/tmp") as tmp:
        chunk_dir = Path(tmp)
        for batch_number, start in enumerate(range(0, len(meta), args.batch_size), start=1):
            batch_meta = meta.iloc[start:start + args.batch_size].copy()
            base.register_stays(connection, batch_meta)
            measurements = base.pull_measurements(connection, args.mimic_dir)
            evidence = None
            if args.include_treatment_context:
                evidence = base.pull_treatment_evidence(connection, args.mimic_dir)
            frame = build_batch_frame(batch_meta, measurements, future_suffix, evidence)
            chunk_path = chunk_dir / f"part-{batch_number:05d}.parquet"
            frame.to_parquet(chunk_path, index=False)
            careunits.update(frame["first_careunit"].dropna().astype(str).tolist())
            counts = base.pair_counts(frame, future_suffix)
            chunk_reports.append({
                "batch": batch_number,
                "stays": int(batch_meta["stay_id"].nunique()),
                "rows": int(len(frame)),
                "pair_counts": counts,
            })
            print(
                f"batch {batch_number}: stays={len(batch_meta):,} "
                f"measurements={len(measurements):,} rows={len(frame):,}",
                flush=True,
            )

        if args.output.exists():
            args.output.unlink()
        chunk_glob = str(chunk_dir / "*.parquet").replace("'", "''")
        output_path = str(args.output).replace("'", "''")
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{chunk_glob}')) "
            f"TO '{output_path}' (FORMAT PARQUET)"
        )

    pair_counts = {
        variable: int(sum(report["pair_counts"].get(variable, 0) for report in chunk_reports))
        for variable in base.STATE_VARS
    }
    rows = int(sum(report["rows"] for report in chunk_reports))
    report = {
        "artifact": "MIMIC-IV v3.1 whole-ICU observation transition cohort (batched)",
        "mimic_dir": str(args.mimic_dir),
        "output": str(args.output.resolve()),
        "horizon_hours": float(args.horizon_hours),
        "future_suffix": future_suffix.lstrip("_"),
        "anchor_step_h": float(base.ANCHOR_STEP_H),
        "lookback_h": float(base.LOOKBACK_H),
        "target_tolerance_h": float(base.TARGET_TOL_H),
        "selected_stays": int(meta["stay_id"].nunique()),
        "subjects": int(meta["subject_id"].nunique()),
        "stays": int(meta["stay_id"].nunique()),
        "rows": rows,
        "careunits": len(careunits),
        "batches": len(chunk_reports),
        "batch_size": int(args.batch_size),
        "pair_counts": dict(sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))),
        "eligible_pair_targets": int(sum(count >= 1 for count in pair_counts.values())),
        "treatment_context": {
            "included": bool(args.include_treatment_context),
            "factual_observed_treatment_only": bool(args.include_treatment_context),
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
        },
        "patient_ids_included_in_report": False,
        "row_level_outputs_committed": False,
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "rows": rows, "batches": len(chunk_reports)}))


if __name__ == "__main__":
    main()

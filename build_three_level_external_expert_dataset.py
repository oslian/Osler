"""Build a MIMIC-trained/eICU-tested three-level expert-anchor dataset.

The externally validated continuous cells become frozen system-layer anchors.
Binary event experts remain in the registry as respiratory event outputs; they
are not mixed into the continuous residual objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import (
    choose_eicu_cohort,
    feature_subset,
    fit_ridge_source,
    mimic_path,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import (
    TARGET_CHANGE_THRESHOLDS,
    _feature_frame,
    _fit_hurdle,
)


CONTINUOUS_TASK = "continuous_value"
EVENT_TASK = "binary_event_probability"


def is_continuous_task(task_type: str) -> bool:
    return str(task_type) != EVENT_TASK


def validated_cells(report: dict) -> list[dict]:
    return [
        {
            "cell": str(item["cell"]),
            "target": str(item["target"]),
            "horizon_hours": int(item["horizon_hours"]),
            "task_type": str(item["task_type"]),
            "source_model": str(item["source_model"]),
            "anchor_column": (
                f"expert__{item['target']}__{int(item['horizon_hours'])}h"
            ),
            "role": (
                "three_level_value_anchor"
                if is_continuous_task(item["task_type"])
                else "retained_respiratory_event_expert"
            ),
        }
        for item in report["cells"]
        if item.get("full_external_pass")
    ]


def fit_frozen_expert(
    frame: pd.DataFrame,
    *,
    target: str,
    source_model: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    train_rows = np.flatnonzero(frame["database"].eq("MIMIC-IV"))
    prediction_rows = np.arange(len(frame), dtype=np.int64)
    features = feature_subset(_feature_frame(frame), source_model, target)
    if source_model == "target_specific_change_hurdle":
        (
            rows,
            prediction,
            _,
            _,
            _,
            selected_policy,
        ) = _fit_hurdle(
            frame,
            features,
            train_rows,
            prediction_rows,
            seed=seed,
            target=target,
            change_threshold=float(TARGET_CHANGE_THRESHOLDS[target]),
        )
        metadata = {
            "source_model": source_model,
            "selected_policy": selected_policy,
        }
    else:
        fitted = fit_ridge_source(
            frame,
            features,
            train_rows,
            prediction_rows,
            target=target,
            source=source_model,
            seed=seed,
        )
        rows = fitted["test_rows"]
        prediction = fitted["test_prediction"]
        metadata = {
            "source_model": source_model,
            "feature_count": fitted["feature_count"],
        }
    output = np.full(len(frame), np.nan, dtype=np.float64)
    output[np.asarray(rows, dtype=np.int64)] = prediction
    return output, metadata


def canonicalize_cell(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    anchor_column: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    keep = []
    for column in frame.columns:
        if column.startswith("future_"):
            continue
        if "_tp" in column or column in {"t_plus"}:
            continue
        keep.append(column)
    output = frame[keep].copy()
    output["horizon_hours"] = float(horizon)
    output["_source_horizon"] = float(horizon)
    output["encounter_id"] = output["stay_id"].astype(str)
    output["anchor_time"] = pd.to_datetime(output["anchor_time"], errors="coerce")
    output[f"future_{target}"] = pd.to_numeric(
        frame[f"future_{target}"], errors="coerce"
    )
    valid_future = np.isfinite(output[f"future_{target}"].to_numpy(dtype=float))
    output[f"future_delta_t_hr_{target}"] = np.where(
        valid_future, float(horizon), np.nan
    )
    output[f"future_label_weight_{target}"] = valid_future.astype(np.float32)
    current_variables = tuple(
        column[:-2]
        for column in output.columns
        if (
            column.endswith("_t")
            and column != "t"
            and f"{column[:-2]}_age_hr" in output.columns
        )
    )
    derived: dict[str, object] = {}
    for variable in current_variables:
        derived[f"_observed_{variable}"] = np.isfinite(
            pd.to_numeric(output[f"{variable}_t"], errors="coerce")
        ).astype(np.int8)
        if variable == target:
            continue
        derived[f"future_{variable}"] = np.nan
        derived[f"future_delta_t_hr_{variable}"] = np.nan
        derived[f"future_label_weight_{variable}"] = np.float32(0.0)
    derived[anchor_column] = np.asarray(prediction, dtype=np.float64)
    return pd.concat(
        (output, pd.DataFrame(derived, index=output.index)), axis=1
    ).copy()


def run(args: argparse.Namespace) -> dict:
    external_report = json.loads(
        args.external_report.read_text(encoding="utf-8")
    )
    cells = validated_cells(external_report)
    if not cells:
        raise ValueError("external report has no full-pass expert cells")

    frames: list[pd.DataFrame] = []
    cell_reports = []
    mimic_cache: dict[int, pd.DataFrame] = {}
    for cell in cells:
        if not is_continuous_task(cell["task_type"]):
            cell_reports.append(
                {
                    **cell,
                    "rows": 0,
                    "status": "retained_outside_continuous_residual_loss",
                }
            )
            continue
        target = cell["target"]
        horizon = cell["horizon_hours"]
        print(f"building {cell['cell']}", flush=True)
        if horizon not in mimic_cache:
            mimic_cache[horizon] = pd.read_parquet(
                mimic_path(args.project_root, horizon)
            )
        mimic, mimic_support = normalize_cohort(
            mimic_cache[horizon],
            database="MIMIC-IV",
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        eicu, eicu_info = choose_eicu_cohort(
            args.eicu_root,
            target=target,
            horizon=horizon,
            max_rows=args.max_rows_per_database,
        )
        if mimic.empty or eicu.empty:
            raise ValueError(f"insufficient data for {cell['cell']}")
        mimic["database"] = "MIMIC-IV"
        eicu["database"] = "eICU"
        combined = pd.concat((mimic, eicu), ignore_index=True, sort=False)
        prediction, fit_metadata = fit_frozen_expert(
            combined,
            target=target,
            source_model=cell["source_model"],
            seed=stable_seed(cell["cell"], "three_level_anchor"),
        )
        canonical = canonicalize_cell(
            combined,
            target=target,
            horizon=horizon,
            anchor_column=cell["anchor_column"],
            prediction=prediction,
        )
        frames.append(canonical)
        cell_reports.append(
            {
                **cell,
                "rows": int(len(canonical)),
                "mimic_rows": int(canonical["database"].eq("MIMIC-IV").sum()),
                "eicu_rows": int(canonical["database"].eq("eICU").sum()),
                "mimic_support": mimic_support,
                "eicu_module": eicu_info.get("module"),
                "fit": fit_metadata,
                "finite_anchor_rows": int(np.isfinite(prediction).sum()),
                "status": "three_level_value_anchor_ready",
            }
        )

    dataset = pd.concat(frames, ignore_index=True, sort=False)
    dataset.sort_values(
        ["database", "subject_id", "anchor_time", "horizon_hours"],
        inplace=True,
    )
    dataset.reset_index(drop=True, inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)

    registry = {
        "schema": "three_level_external_expert_anchor_registry.v1",
        "source_external_report": str(args.external_report.resolve()),
        "cells": cells,
        "continuous_value_anchor_cells": sum(
            is_continuous_task(cell["task_type"]) for cell in cells
        ),
        "retained_event_expert_cells": sum(
            cell["task_type"] == EVENT_TASK for cell in cells
        ),
        "binary_event_outputs_mixed_into_continuous_loss": False,
        "expert_layer": "system",
        "legal_residual_path": "system <-> organ <-> body",
        "causal_claim_allowed": False,
    }
    args.registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    report = {
        "schema": "three_level_external_expert_dataset_report.v1",
        "dataset": str(args.output.resolve()),
        "registry": str(args.registry.resolve()),
        "rows": int(len(dataset)),
        "subjects": int(dataset["subject_id"].nunique()),
        "mimic_rows": int(dataset["database"].eq("MIMIC-IV").sum()),
        "eicu_rows": int(dataset["database"].eq("eICU").sum()),
        "cells": cell_reports,
        "leakage_boundary": (
            "experts fit on MIMIC only; eICU labels used only by downstream "
            "external scoring"
        ),
        "status": "ready_for_bounded_three_level_external_test",
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-report",
        type=Path,
        default=Path("mimic_to_eicu_49_cell_external_audit_20260816.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--eicu-root", type=Path, default=Path("eicu_future_label_repaired")
    )
    parser.add_argument("--max-rows-per-database", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "rows": report["rows"],
                "subjects": report["subjects"],
                "status": report["status"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

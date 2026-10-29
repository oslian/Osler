"""Simulated deployment of the binary event cells with online calibration (development databases).

The frozen static hypoxemia model is fitted on each of 5 MIMIC samples (as
``evaluate_event`` fits it) and every external hospital is replayed as its own
live deployment on the shared deployment clock (``site_adaptive_interval_dev.replay``):
online Platt scaling (``site_adaptive_event_amendment_20261028.json``) starts
from the frozen model and learns from outcomes only once they resolve; warm-up
stays are not scored.  The MIMIC-prevalence comparator receives the same
recalibration; persistence (current SpO2 <= 92) is unchanged.

Gates (``site_adaptive_event_dev.score_stream``): Brier beats persistence and
the comparator on the point axes of ``GATE_AXES`` (patient; plus hospital on
eICU), and calibration over the scored stream passes
``osler_jepa.event_calibration`` (CITL in [-0.2, 0.2], slope in [0.8, 1.25]).
Robust = full pass in >= 80% of samples.  Static runs on the same scored rows.
The calibration gate is pooled over the stream: a site-level claim.  Not gated,
for interpretation: patient-cluster bootstrap intervals and per-group estimates
with support counts by hospital, care unit and stream phase (thirds of each
hospital's clock order).  The development databases are in-sample for the
post-hoc choice of the method.  Results are a retrospective observation-time
replay, not a prospective availability claim.

A lockbox database (SICdb, HiRID) is scored only under a frozen protocol whose
scored task types include binary event probabilities (``lockbox_step``), run
from the project folder: every hashed file unchanged; every project module
loaded from this project and hashed; the event horizons' MIMIC training files
hashed; the scored cells exactly the protocol's ``scored_cells``; the
invocation equal to its ``deployment_form_arguments``; an addendum receipt that
recorded the event horizons' extraction files unchanged and byte-identical to
the ``previous_protocol``'s addendum; then the one-shot ledger step
``binary_<db>`` is claimed, and only then can ``external_cohort`` read the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import (
    EXTERNAL_PREFIX, GATE_AXES, LOCKBOX_DATABASES, evaluability, external_cohort, verify_protocol,
)
from mimic_to_eicu_49_cell_external_audit import normalize_cohort, stable_seed
from multi_sample_external_audit import ROBUST_SHARE, mimic_sample
from osler_jepa.event_calibration import cluster_bootstrap, group_calibration
from osler_jepa.frozen_protocol import claim_step, record_step, sha256, step_record
from osler_jepa.online_platt import OnlinePlattConfig
from site_adaptive_event_dev import THRESHOLD, recalibrate, score_stream, static_model
from site_adaptive_interval_dev import replay
from target_specific_domain_conformal_audit import _safe_float

TASK_TYPE = "binary_event_probability"
METHODS = ("static", "online_platt")
PHASES = 3
FORM_ARGUMENTS = ("mimic_template", "development_report", "event_config", "samples", "max_rows_per_database")
PROJECT = Path(__file__).resolve().parent


def form_arguments(args: argparse.Namespace) -> dict:
    """The invocation that defines the scored deployment form, as the protocol records it."""

    return {name: str(value) if isinstance(value, Path) else value for name, value in ((name, getattr(args, name)) for name in FORM_ARGUMENTS)}


def event_contracts(path: Path) -> list[dict]:
    development = json.loads(path.read_text(encoding="utf-8"))
    return sorted((row for row in development["cells"] if row["task_type"] == TASK_TYPE), key=lambda row: int(row["horizon_hours"]))


def project_module_name(name: str) -> bool:
    top = name.split(".")[0]
    return (PROJECT / f"{top}.py").exists() or (PROJECT / top / "__init__.py").exists()


def loaded_project_files() -> set[str]:
    """Project source files loaded in this process, relative to the project (the protocol's path convention).

    A module named like a project module but loaded from elsewhere, or a project file loaded from anything but
    its ``.py`` source, raises.  Other relative pseudo-paths (torch.ops, torch.classes) are not project files."""

    files = set()
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file)
        inside = path.is_absolute() and path.resolve().is_relative_to(PROJECT) and ".venv" not in path.resolve().parts
        if project_module_name(name) and not inside:
            raise RuntimeError(f"project module {name} is not loaded from {PROJECT}: {file}")
        if inside:
            if path.suffix != ".py":
                raise RuntimeError(f"project module {name} is not loaded from its source: {file}")
            files.add(str(path.resolve().relative_to(PROJECT)))
    return files


def lockbox_step(args: argparse.Namespace) -> str | None:
    """For a lockbox: verify the frozen protocol, its scope and invocation, the addendum, and claim ``binary_<db>``."""

    if args.database not in LOCKBOX_DATABASES:
        return None
    if args.protocol is None:
        raise ValueError(f"{args.database} is an untouched lockbox: --protocol is required")
    if not args.external_template:
        raise ValueError("--external-template is required for a lockbox database")
    if Path.cwd().resolve() != PROJECT:
        raise RuntimeError(f"lockbox scoring runs from the project folder {PROJECT}, where the protocol's paths resolve")
    protocol = verify_protocol(args.protocol)
    if TASK_TYPE not in protocol.get("scored_task_types", []):
        raise RuntimeError("binary event cells are outside this protocol's lockbox scope")
    contracts = event_contracts(args.development_report)
    if sorted(row["cell"] for row in contracts) != sorted(protocol.get("scored_cells", [])):
        raise RuntimeError("event cells differ from the protocol's scored_cells")
    previous = protocol.get("previous_protocol")
    required = loaded_project_files() | {str(args.event_config), str(args.development_report)} | ({previous} if previous else set())
    unhashed = sorted(required - set(protocol["frozen_code_sha256"]))
    if unhashed:
        raise RuntimeError(f"files used by the scoring are not hashed by the protocol: {unhashed}")
    horizons = sorted({int(row["horizon_hours"]) for row in contracts})
    untrained = sorted({args.mimic_template.format(horizon=h) for h in horizons} - set(protocol.get("frozen_training_data_sha256", {})))
    if untrained:
        raise RuntimeError(f"MIMIC training files not hashed by the protocol: {untrained}")
    if form_arguments(args) != protocol["deployment_form_arguments"]:
        raise RuntimeError("lockbox invocation differs from the protocol's frozen deployment form")
    addendum = step_record(args.protocol, f"addendum_{args.database}")      # raises if no addendum was recorded
    changed = sorted(name for name, digest in addendum["inputs_sha256"].items() if sha256(Path(name)) != digest)
    if changed:
        raise RuntimeError(f"extraction files changed after the addendum: {changed}")
    templates = sorted({args.external_template.format(horizon=h) for h in horizons})
    unrecorded = [name for name in templates if name not in addendum["inputs_sha256"]]
    if unrecorded:
        raise RuntimeError(f"external transitions not recorded in the addendum: {unrecorded}")
    if previous:
        prior = step_record(Path(previous), f"addendum_{args.database}")["inputs_sha256"]
        differing = [name for name in templates if prior.get(name) != addendum["inputs_sha256"][name]]
        if differing:
            raise RuntimeError(f"extraction files differ from the previous protocol's addendum: {differing}")
    step = f"binary_{args.database}"
    claim_step(args.protocol, step)
    args.claimed_step = step                                 # external_cohort reads lockbox data only after this claim
    return step


def frozen_event_config(path: Path) -> OnlinePlattConfig:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["selected"] != "online_platt":
        raise ValueError(f"online Platt scaling was not selected in {path}")
    config = record["config"]
    if config["initial"] != [1.0, 0.0]:
        raise ValueError("online Platt scaling must start from the frozen model")
    return OnlinePlattConfig(gamma=config["gamma"], rho=config["rho"], radius=config["radius"])


def simulate_sample(frame: pd.DataFrame, model: dict[str, pd.Series], *, cell: str, sample: int,
                    config: OnlinePlattConfig) -> dict[str, np.ndarray]:
    """Each hospital as its own deployment; returns rows, observed, scored mask and both methods' predictions."""

    test = model["probability"].index.to_numpy()
    hospitals = frame.iloc[test]["hospitalid"].astype(str).to_numpy()
    parts = []
    for hospital in np.unique(hospitals):
        deployment = replay(frame, test[hospitals == hospital], "o2sat", stable_seed(cell, sample, hospital, "clock"))
        rows = deployment.rows
        observed = (_safe_float(frame, "future_o2sat")[rows] <= THRESHOLD).astype(float)
        probability, prevalence = model["probability"][rows].to_numpy(), model["prevalence"][rows].to_numpy()
        parts.append({
            "rows": rows, "observed": observed, "scored": ~deployment.warm,
            "phase": np.minimum(PHASES * np.arange(len(rows)) // len(rows), PHASES - 1),
            "persistence": model["persistence"][rows].to_numpy(),
            "static": probability, "static_comparator": prevalence,
            "online_platt": recalibrate(deployment, probability, observed, config),
            "online_platt_comparator": recalibrate(deployment, prevalence, observed, config),
        })
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def diagnostics(frame: pd.DataFrame, stream: dict[str, np.ndarray], keep: np.ndarray, label: str, *, seed: int) -> dict:
    """Not gated: patient-cluster bootstrap intervals and per-group calibration with support counts."""

    rows = stream["rows"][keep]
    predictions = pd.DataFrame({"subject_id": frame.iloc[rows]["subject_id"].astype(str).to_numpy(),
                                "hospitalid": frame.iloc[rows]["hospitalid"].astype(str).to_numpy(),
                                "careunit": frame.iloc[rows]["careunit"].astype(str).to_numpy(),
                                "phase": stream["phase"][keep],
                                "probability": stream[label][keep], "outcome": stream["observed"][keep]})
    return {"patient_bootstrap": cluster_bootstrap(predictions, "subject_id", seed=seed),
            **{f"by_{column}": group_calibration(predictions, column) for column in ("hospitalid", "careunit", "phase")}}


def summarize(samples: list[dict], label: str) -> dict:
    share = float(np.mean([s[label]["full_pass"] for s in samples]))
    def spread(key):
        values = [s[label][key] for s in samples if s[label].get(key) is not None]
        return [min(values), max(values)] if values else None

    return {
        "full_pass_share": share,
        "point_pass_share": float(np.mean([s[label]["point_pass"] for s in samples])),
        "calibration_pass_share": float(np.mean([s[label]["calibration_pass"] for s in samples])),
        "invalid_samples": sum(bool(s[label]["invalid_predictions"]) for s in samples),
        **{f"{key}_range": spread(key) for key in ("citl", "slope", "brier")},
        "robust": share >= ROBUST_SHARE,
    }


def run(args: argparse.Namespace) -> dict:
    step = lockbox_step(args)
    config = frozen_event_config(args.event_config)
    contracts = event_contracts(args.development_report)
    axes = GATE_AXES[args.database]["point"]
    cells = []
    for contract in contracts:
        cell, target, horizon = contract["cell"], contract["target"], int(contract["horizon_hours"])
        external, _ = external_cohort(args, target, horizon)
        reason = evaluability(args.database, external)
        if reason:
            cells.append({"cell": cell, "evaluable": False, "reason": reason})
            continue
        full, _ = normalize_cohort(pd.read_parquet(args.mimic_template.format(horizon=horizon)), database="MIMIC-IV",
                                   target=target, horizon=horizon, max_rows=0)
        samples = []
        for sample in range(args.samples):
            frame = pd.concat([mimic_sample(full, target=target, horizon=horizon, sample=sample, rows=15000), external],
                              ignore_index=True, sort=False)
            frame.sort_values(["subject_id", "anchor_time"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
            subject = frame["subject_id"].astype(str)
            model = static_model(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                 np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX[args.database])), cell)
            stream = simulate_sample(frame, model, cell=cell, sample=sample, config=config)
            keep = stream["scored"]
            entry = {"sample": sample, "scored_rows": int(keep.sum())}
            for label in METHODS:
                entry[label] = score_stream(frame, stream["rows"][keep], stream["observed"][keep], stream[label][keep],
                                            stream["persistence"][keep], stream[f"{label}_comparator"][keep],
                                            axes=axes, seed=stable_seed(cell, sample, "brier"))
                if not entry[label]["invalid_predictions"]:
                    entry[label]["diagnostics"] = diagnostics(frame, stream, keep, label, seed=stable_seed(cell, sample, label, "diagnostics"))
            samples.append(entry)
        methods = {label: summarize(samples, label) for label in METHODS}
        cells.append({"cell": cell, "evaluable": True, "rows": int(len(model["probability"])),
                      "hospitals": int(external["hospitalid"].nunique()), "methods": methods, "per_sample": samples})
        print(json.dumps({"cell": cell, **{label: {k: v for k, v in m.items() if k != "robust"} for label, m in methods.items()}}), flush=True)
    evaluable = [cell for cell in cells if cell["evaluable"]]
    output = {
        "schema": "site_adaptive_event_audit.v2",
        "external_database": args.database,
        "mimic_template": args.mimic_template,
        "event_config": str(args.event_config),
        "samples": args.samples,
        "external_rows_per_cell": args.max_rows_per_database,
        "robust_counts": {label: sum(cell["methods"][label]["robust"] for cell in evaluable) for label in METHODS},
        "evaluable_event_cells": len(evaluable),
        "independent_validation": False,
        "claim_scope": "retrospective observation-time replay; pooled site-level calibration",
        "cells": cells,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if step:
        inputs = [args.external_template.format(horizon=int(row["horizon_hours"])) for row in contracts]
        record_step(args.protocol, step, inputs, [str(args.output)])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(GATE_AXES), required=True)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--external-template", default=None, help="parquet path template with {horizon} for single-site databases")
    parser.add_argument("--max-rows-per-database", type=int, default=60000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--event-config", type=Path, default=Path("site_adaptive_event_amendment_20261028.json"))
    parser.add_argument("--protocol", type=Path, default=None, help="frozen lockbox protocol; required for lockbox databases (SICdb, HiRID)")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""Validate physiological subtype adaptation from eICU into MIMIC-IV.

The subtype router and target residual priors are fitted only on eICU. MIMIC
labels are a lockbox used exclusively for scoring. Each continuous runtime cell
compares the current release point forecast against a physiological-subtype
adapter and a capacity-matched patient-shuffled subtype placebo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.runtime import (
    _multiscale_history_indices,
    _predict_head,
    load_release,
)
from whole_body_jepa.subtypes import (
    SoftPhysiologySubtypeModel,
    causal_history_covariates_at_anchors,
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def mimic_path(project_root: Path, horizon: int) -> Path:
    name = (
        f"mimiciv_observation_transitions_{horizon}h_"
        "full_v31_treatment_unit_safe.parquet"
    )
    temporary = Path("/private/tmp") / name
    return temporary if temporary.exists() else project_root / name


def eicu_path(root: Path, target: str, horizon: int) -> tuple[Path | None, str | None]:
    """Choose a canonical discovery source by hospital and patient coverage.

    Disease-name preference previously selected narrow 12-hospital cohorts for
    common targets such as MAP even though the same measured target existed in
    200+ hospitals. Source choice is now target support, not disease naming.
    """

    import pyarrow.parquet as pq

    candidates: list[tuple[tuple[int, int, int, int, int], Path, str]] = []
    suffix = f"_transitions_{horizon}h.parquet"
    for path in sorted(root.glob(f"eicu_*{suffix}")):
        columns = set(pq.read_schema(path).names)
        required = {
            "stay_id",
            "hospitalid",
            f"{target}_t",
            f"{target}_tp{horizon}",
        }
        if not required.issubset(columns):
            continue
        read_columns = [
            "stay_id",
            "hospitalid",
            f"{target}_t",
            f"{target}_tp{horizon}",
        ]
        quality_column = f"future_label_quality_{target}"
        weight_column = f"future_label_weight_{target}"
        read_columns.extend(
            column
            for column in (quality_column, weight_column)
            if column in columns
        )
        coverage = pd.read_parquet(path, columns=read_columns)
        valid = (
            pd.to_numeric(coverage[f"{target}_t"], errors="coerce").notna()
            & pd.to_numeric(
                coverage[f"{target}_tp{horizon}"], errors="coerce"
            ).notna()
        )
        if quality_column in coverage:
            valid &= coverage[quality_column].astype(str).str.lower().isin(
                {"exact", "near"}
            )
        if weight_column in coverage:
            valid &= pd.to_numeric(
                coverage[weight_column], errors="coerce"
            ).fillna(1.0).gt(0.0)
        coverage = coverage.loc[valid, ["stay_id", "hospitalid"]]
        hospital_depth = coverage.groupby("hospitalid")["stay_id"].nunique()
        score = (
            int((hospital_depth >= 3).sum()),
            int((hospital_depth >= 10).sum()),
            int((hospital_depth >= 5).sum()),
            int(coverage["stay_id"].nunique()),
            int(len(coverage)),
        )
        module = path.name.removeprefix("eicu_").removesuffix(suffix)
        candidates.append((score, path, module))
    if not candidates:
        return None, None
    _, path, module = max(candidates, key=lambda item: item[0])
    return path, module


def _eicu_read_columns(
    path: Path,
    *,
    target: str,
    horizon: int,
    variables: tuple[str, ...],
) -> list[str]:
    """Read only columns used by JEPA inference and pre-anchor adaptation."""

    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    requested = {
        "subject_id",
        "stay_id",
        "hospitalid",
        "unittype",
        "t",
        "onset_hour",
        "hours_since_onset",
        f"{target}_t",
        f"{target}_tp{horizon}",
        f"future_label_quality_{target}",
        f"future_label_weight_{target}",
    }
    for variable in variables:
        requested.add(f"{variable}_t")
        requested.add(f"{variable}_age_hr")
    requested.update(column for column in available if column.startswith("hist_"))
    return sorted(requested & available)


def _relative_time(frame: pd.DataFrame) -> np.ndarray:
    if "hours_since_onset" in frame:
        values = pd.to_numeric(frame["hours_since_onset"], errors="coerce")
        if values.notna().any():
            filled = values.groupby(frame["stay_id"], sort=False).transform(
                lambda group: group.interpolate().ffill().bfill()
            )
            return filled.fillna(0.0).to_numpy(dtype=np.float64)
    timestamp = pd.to_datetime(frame["t"], errors="coerce")
    origin = timestamp.groupby(frame["stay_id"], sort=False).transform("min")
    return ((timestamp - origin).dt.total_seconds() / 3600.0).fillna(0.0).to_numpy()


def _sample_label_rows(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    max_rows: int,
    max_rows_per_subject: int,
    seed: int,
) -> pd.DataFrame:
    current = pd.to_numeric(frame[f"{target}_t"], errors="coerce")
    future = pd.to_numeric(frame[f"{target}_tp{horizon}"], errors="coerce")
    valid = np.isfinite(current) & np.isfinite(future)
    quality = f"future_label_quality_{target}"
    if quality in frame:
        valid &= frame[quality].astype(str).str.lower().isin({"exact", "near"})
    weight = f"future_label_weight_{target}"
    if weight in frame:
        numeric_weight = pd.to_numeric(frame[weight], errors="coerce")
        valid &= numeric_weight.fillna(1.0).gt(0.0)
    view = frame.loc[valid].copy()
    if view.empty:
        return view
    view["subject_id"] = view["subject_id"].astype(str)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    subjects = view["subject_id"].unique().astype(object)
    rng.shuffle(subjects)
    for subject in subjects:
        rows = view.index[view["subject_id"].eq(subject)].to_numpy()
        order = np.argsort(pd.to_datetime(view.loc[rows, "t"], errors="coerce"))
        rows = rows[order]
        if len(rows) > max_rows_per_subject:
            positions = np.linspace(0, len(rows) - 1, max_rows_per_subject)
            rows = rows[np.round(positions).astype(int)]
        selected.extend(rows.tolist())
        if len(selected) >= max_rows:
            break
    view = frame.loc[selected[:max_rows]].copy()
    view.sort_values(["stay_id", "t"], kind="stable", inplace=True)
    return view.reset_index(drop=True)


def _model_inputs(frame: pd.DataFrame, assets: Any, horizon: int) -> dict[str, np.ndarray]:
    variables = tuple(assets.checkpoint["variables"])
    model_frame = frame.copy()
    model_frame["stay_id"] = model_frame["stay_id"].astype(str)
    model_frame["_time_hours"] = _relative_time(model_frame)
    model_frame["hours_since_onset"] = model_frame["_time_hours"]
    for variable in variables:
        current = f"{variable}_t"
        age = f"{variable}_age_hr"
        if current not in model_frame:
            model_frame[current] = np.nan
        if age not in model_frame:
            model_frame[age] = np.nan
    current, _, _ = assets.scaler.normalize_current(model_frame)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.ones_like(current, dtype=np.float32)
    ages = np.zeros_like(current, dtype=np.float32)
    presence = np.ones(
        (len(model_frame), len(assets.checkpoint["modules"])), dtype=np.float32
    )
    onset = np.clip(model_frame["hours_since_onset"].to_numpy(), 0.0, 168.0)[:, None]
    onset = (onset / 24.0).astype(np.float32)
    origin = model_frame.groupby("stay_id", sort=False)["_time_hours"].transform("min")
    elapsed = np.clip(model_frame["_time_hours"] - origin, 0.0, 168.0)
    elapsed = (elapsed.to_numpy(dtype=np.float32)[:, None] / 168.0)
    matrix = np.concatenate(
        (current, mask, ages, onset, presence, elapsed), axis=1
    ).astype(np.float32)
    if matrix.shape[1] != int(assets.checkpoint["input_dim"]):
        raise ValueError(f"unexpected model input width: {matrix.shape[1]}")
    return {
        "frame": model_frame,
        "current": current,
        "mask": mask,
        "ages": ages,
        "presence": presence,
        "matrix": matrix,
        "history": _multiscale_history_indices(model_frame),
        "horizon": np.full(len(model_frame), float(horizon), dtype=np.float32),
    }


def _base_and_features(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    cell_contract: dict[str, Any],
    assets: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _model_inputs(frame, assets, horizon)
    variables = tuple(assets.checkpoint["variables"])
    target_index = variables.index(target)
    module_index = int(assets.model.variable_module_ids[target_index].item())
    features: list[np.ndarray] = []
    model_points: list[np.ndarray] = []
    assets.model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            rows = np.arange(start, min(start + batch_size, len(frame)))
            output = assets.model(
                torch.from_numpy(arrays["matrix"][arrays["history"][rows]]).float(),
                torch.from_numpy(arrays["current"][rows]).float(),
                torch.from_numpy(arrays["mask"][rows]).float(),
                torch.from_numpy(arrays["ages"][rows]).float(),
                torch.from_numpy(arrays["presence"][rows]).float(),
                torch.from_numpy(arrays["horizon"][rows]).float(),
            )
            normalized = output["value"].cpu().numpy().astype(np.float64)
            physical = normalized * assets.scaler.scales + assets.scaler.medians
            current = arrays["current"][rows]
            block = np.concatenate(
                (
                    output["context_latent"].cpu().numpy(),
                    output["belief_state"].cpu().numpy(),
                    output["predicted_target_latent"][:, target_index].cpu().numpy(),
                    output["organ_latents"][:, module_index].cpu().numpy(),
                    normalized[:, target_index : target_index + 1],
                    current,
                    current[:, target_index : target_index + 1],
                ),
                axis=1,
            ).astype(np.float32)
            features.append(block)
            model_points.append(physical[:, target_index])
    feature_matrix = np.concatenate(features, axis=0)
    model_point = np.concatenate(model_points, axis=0)
    current_physical = pd.to_numeric(frame[f"{target}_t"], errors="coerce").to_numpy()
    cell = str(cell_contract["cell"])
    if cell in set(assets.heads["externally_validated_heads"]):
        head = assets.heads["heads"][cell]
        base = np.asarray(
            [
                _predict_head(head, feature_matrix[row : row + 1], current_physical[row])
                for row in range(len(frame))
            ],
            dtype=np.float64,
        )
    elif bool(cell_contract.get("can_move")):
        base = model_point
    else:
        base = current_physical.astype(np.float64)
    return feature_matrix, base


def _load_cell(
    *,
    database: str,
    project_root: Path,
    eicu_root: Path,
    target: str,
    horizon: int,
    cell_contract: dict[str, Any],
    assets: Any,
    max_rows: int,
    max_rows_per_subject: int,
    batch_size: int,
    include_complete_history: bool = False,
) -> dict[str, Any] | None:
    if database == "eICU" and include_complete_history:
        path, module = eicu_path(eicu_root, target, horizon)
        if path is None:
            return None
        columns = _eicu_read_columns(
            path,
            target=target,
            horizon=horizon,
            variables=tuple(assets.checkpoint["variables"]),
        )
        frame = pd.read_parquet(path, columns=columns)
    else:
        path = mimic_path(project_root, horizon)
        module = None
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
    required = {"subject_id", "stay_id", "t", f"{target}_t", f"{target}_tp{horizon}"}
    if not required.issubset(frame.columns):
        return None
    complete_frame = frame
    frame = _sample_label_rows(
        complete_frame,
        target=target,
        horizon=horizon,
        max_rows=max_rows,
        max_rows_per_subject=max_rows_per_subject,
        seed=stable_seed(database, target, horizon),
    )
    if len(frame) < 100 or frame["subject_id"].nunique() < 40:
        return None
    prefix = "EICU" if database == "eICU" else "MIMIC"
    frame["trajectory_group"] = prefix + "-STAY:" + frame["stay_id"].astype(str)
    pre_anchor_history = None
    if database == "eICU":
        history = complete_frame[
            complete_frame["stay_id"].isin(frame["stay_id"].unique())
        ].copy()
        history_groups = prefix + "-STAY:" + history["stay_id"].astype(str)
        variables = tuple(assets.checkpoint["variables"])
        raw = np.column_stack(
            [
                pd.to_numeric(history.get(f"{variable}_t"), errors="coerce").to_numpy()
                if f"{variable}_t" in history
                else np.full(len(history), np.nan)
                for variable in variables
            ]
        ).astype(np.float64)
        observed = np.isfinite(raw)
        normalized = np.divide(
            raw - assets.scaler.medians,
            assets.scaler.scales,
            out=np.zeros_like(raw),
            where=observed,
        )
        ages = np.column_stack(
            [
                pd.to_numeric(
                    history.get(f"{variable}_age_hr"), errors="coerce"
                ).to_numpy()
                if f"{variable}_age_hr" in history
                else np.full(len(history), np.nan)
                for variable in variables
            ]
        ).astype(np.float64)
        pre_anchor_history = causal_history_covariates_at_anchors(
            normalized,
            observed_mask=observed,
            measurement_ages=ages,
            history_group_ids=history_groups.to_numpy(dtype=object),
            history_times=_relative_time(history),
            anchor_group_ids=frame["trajectory_group"].to_numpy(dtype=object),
            anchor_times=_relative_time(frame),
            window_hours=float(horizon),
        )
    frame["subject_key"] = prefix + ":" + frame["subject_id"].astype(str)
    frame["careunit"] = (
        frame.get(
            "unittype" if database == "eICU" else "first_careunit",
            pd.Series("unknown", index=frame.index),
        )
        .fillna("unknown")
        .astype(str)
    )
    frame["hospital"] = (
        frame.get(
            "hospitalid" if database == "eICU" else "first_careunit",
            pd.Series("unknown", index=frame.index),
        )
        .fillna("unknown")
        .astype(str)
    )
    frame["anchor_time"] = pd.to_datetime(frame["t"], errors="coerce")
    features, base = _base_and_features(
        frame,
        target=target,
        horizon=horizon,
        cell_contract=cell_contract,
        assets=assets,
        batch_size=batch_size,
    )
    return {
        "frame": frame,
        "features": features,
        "base": base,
        "truth": pd.to_numeric(
            frame[f"{target}_tp{horizon}"], errors="coerce"
        ).to_numpy(dtype=np.float64),
        "subjects": frame["subject_key"].to_numpy(dtype=object),
        "trajectory_groups": frame["trajectory_group"].to_numpy(dtype=object),
        "careunits": frame["careunit"].to_numpy(dtype=object),
        "hospitals": frame["hospital"].to_numpy(dtype=object),
        "time": frame["anchor_time"].to_numpy(),
        "time_hours": _relative_time(frame),
        "module": module,
        "path": str(path),
        "pre_anchor_history_covariates": pre_anchor_history,
    }


def _ragged_training(cells: list[str], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = sum(len(data[cell]["truth"]) for cell in cells)
    feature_matrix = np.concatenate([data[cell]["features"] for cell in cells])
    base = np.zeros((rows, len(cells)), dtype=np.float32)
    truth = np.zeros_like(base)
    mask = np.zeros_like(base, dtype=bool)
    subjects = np.concatenate([data[cell]["subjects"] for cell in cells])
    offset = 0
    for index, cell in enumerate(cells):
        count = len(data[cell]["truth"])
        base[offset : offset + count, index] = data[cell]["base"]
        truth[offset : offset + count, index] = data[cell]["truth"]
        mask[offset : offset + count, index] = True
        offset += count
    return {
        "features": feature_matrix,
        "base": base,
        "truth": truth,
        "mask": mask,
        "subjects": subjects,
    }


def _group_delta(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    delta = np.empty(len(unique), dtype=np.float64)
    for index in range(len(unique)):
        rows = inverse == index
        delta[index] = np.mean(np.abs(left[rows] - truth[rows])) - np.mean(
            np.abs(right[rows] - truth[rows])
        )
    return unique, delta


def _bootstrap_ci(delta: np.ndarray, *, seed: int, repeats: int) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64)
    if len(delta) < 2:
        return {"groups": int(len(delta)), "mean": None, "ci95": None, "pass": False}
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        draws[index] = rng.choice(delta, size=len(delta), replace=True).mean()
    ci = np.quantile(draws, (0.025, 0.975))
    return {
        "groups": int(len(delta)),
        "mean": float(delta.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "pass": bool(ci[1] < 0.0),
    }


def _gate_pair(
    payload: dict[str, Any],
    candidate: np.ndarray,
    comparator: np.ndarray,
    *,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    truth = payload["truth"]
    _, patient_delta = _group_delta(
        truth, candidate, comparator, payload["subjects"]
    )
    _, careunit_delta = _group_delta(
        truth, candidate, comparator, payload["careunits"]
    )
    timestamp = pd.Series(pd.to_datetime(payload["time"], errors="coerce"))
    if timestamp.notna().any():
        boundary = timestamp.dropna().quantile(0.8)
        late = timestamp.ge(boundary).to_numpy(dtype=bool)
    else:
        late = np.ones(len(truth), dtype=bool)
    _, time_delta = _group_delta(
        truth[late], candidate[late], comparator[late], payload["subjects"][late]
    )
    return {
        "patient": _bootstrap_ci(
            patient_delta, seed=stable_seed(label, "patient"), repeats=repeats
        ),
        "careunit": _bootstrap_ci(
            careunit_delta, seed=stable_seed(label, "careunit"), repeats=repeats
        ),
        "forward_time": _bootstrap_ci(
            time_delta, seed=stable_seed(label, "time"), repeats=repeats
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir)
    contracts = [
        cell
        for cell in assets.registry["cells"]
        if str(cell["target"]) != "hypoxemia_event"
    ]
    eicu: dict[str, dict[str, Any]] = {}
    mimic: dict[str, dict[str, Any]] = {}
    skipped = []
    for index, contract in enumerate(contracts, start=1):
        cell = str(contract["cell"])
        target = str(contract["target"])
        horizon = int(contract["horizon_hours"])
        print(f"[{index}/{len(contracts)}] extracting {cell}", flush=True)
        discovery = _load_cell(
            database="eICU",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target=target,
            horizon=horizon,
            cell_contract=contract,
            assets=assets,
            max_rows=args.discovery_rows_per_cell,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
        )
        external = _load_cell(
            database="MIMIC-IV",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target=target,
            horizon=horizon,
            cell_contract=contract,
            assets=assets,
            max_rows=args.external_rows_per_cell,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
        )
        if discovery is None or external is None:
            skipped.append(
                {
                    "cell": cell,
                    "reason": "insufficient_eicu_or_mimic_support",
                }
            )
            continue
        eicu[cell] = discovery
        mimic[cell] = external

    cells = sorted(set(eicu) & set(mimic))
    if not cells:
        raise ValueError("no cells have both eICU discovery and MIMIC lockbox data")
    training = _ragged_training(cells, eicu)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    predictions: dict[str, dict[str, list[np.ndarray]]] = {
        cell: {"candidate": [], "placebo": []} for cell in cells
    }
    models = []
    for seed_index, seed in enumerate(seeds, start=1):
        print(f"[{seed_index}/{len(seeds)}] fitting subtype and placebo seed {seed}", flush=True)
        candidate = SoftPhysiologySubtypeModel(
            subtype_count=args.subtypes,
            min_effective_subjects=args.min_effective_subjects,
            subtype_prior_strength=args.subtype_prior_strength,
            max_rows_per_subject=args.subtype_rows_per_subject,
            embedding_dim=args.embedding_dim,
            random_state=seed,
        ).fit(
            training["features"],
            base_points=training["base"],
            observed_targets=training["truth"],
            observed_mask=training["mask"],
            subject_ids=training["subjects"],
            cell_keys=cells,
        )
        placebo = SoftPhysiologySubtypeModel(
            subtype_count=args.subtypes,
            min_effective_subjects=args.min_effective_subjects,
            subtype_prior_strength=args.subtype_prior_strength,
            max_rows_per_subject=args.subtype_rows_per_subject,
            embedding_dim=args.embedding_dim,
            random_state=seed,
        ).fit(
            training["features"],
            base_points=training["base"],
            observed_targets=training["truth"],
            observed_mask=training["mask"],
            subject_ids=training["subjects"],
            cell_keys=cells,
            responsibility_permutation_seed=stable_seed(seed, "placebo"),
        )
        models.append(candidate)
        for cell in cells:
            base = mimic[cell]["base"]
            features = mimic[cell]["features"]
            predictions[cell]["candidate"].append(
                base + candidate.subtype_corrections(features, cell=cell)
            )
            predictions[cell]["placebo"].append(
                base + placebo.subtype_corrections(features, cell=cell)
            )

    cell_results = []
    promotable = []
    for cell in cells:
        payload = mimic[cell]
        base = payload["base"]
        truth = payload["truth"]
        candidate_by_seed = predictions[cell]["candidate"]
        placebo_by_seed = predictions[cell]["placebo"]
        candidate = np.mean(candidate_by_seed, axis=0)
        placebo = np.mean(placebo_by_seed, axis=0)
        seed_deltas = []
        for seed, candidate_seed, placebo_seed in zip(
            seeds, candidate_by_seed, placebo_by_seed
        ):
            _, vs_base = _group_delta(
                truth, candidate_seed, base, payload["subjects"]
            )
            _, vs_placebo = _group_delta(
                truth, candidate_seed, placebo_seed, payload["subjects"]
            )
            seed_deltas.append(
                {
                    "seed": seed,
                    "patient_equal_delta_vs_base": float(vs_base.mean()),
                    "patient_equal_delta_vs_placebo": float(vs_placebo.mean()),
                    "pass": bool(vs_base.mean() < 0.0 and vs_placebo.mean() < 0.0),
                }
            )
        vs_base = _gate_pair(
            payload,
            candidate,
            base,
            label=f"{cell}:base",
            repeats=args.bootstrap_repeats,
        )
        vs_placebo = _gate_pair(
            payload,
            candidate,
            placebo,
            label=f"{cell}:placebo",
            repeats=args.bootstrap_repeats,
        )
        pass_all = bool(
            all(item["pass"] for item in seed_deltas)
            and all(gate["pass"] for gate in vs_base.values())
            and all(gate["pass"] for gate in vs_placebo.values())
        )
        if pass_all:
            promotable.append(cell)
        cell_results.append(
            {
                "cell": cell,
                "rows": int(len(truth)),
                "patients": int(len(np.unique(payload["subjects"]))),
                "careunits": int(len(np.unique(payload["careunits"]))),
                "base_mae": float(np.mean(np.abs(base - truth))),
                "subtype_mae": float(np.mean(np.abs(candidate - truth))),
                "placebo_mae": float(np.mean(np.abs(placebo - truth))),
                "seed_results": seed_deltas,
                "subtype_vs_base": vs_base,
                "subtype_vs_placebo": vs_placebo,
                "external_pass": pass_all,
                "runtime_status": "eligible_for_integration" if pass_all else "candidate_only",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if models:
        models[0].save(args.output_dir / "physiology_subtype_seed0.joblib")
    result = {
        "schema": "whole_body_jepa.subtype_external_validation.v1",
        "discovery_database": "eICU",
        "external_lockbox_database": "MIMIC-IV",
        "base_model": "2026-08-22 shared whole-body JEPA release runtime point",
        "candidate": "soft physiological subtype residual adapter",
        "placebo": "patient-level shuffled subtype responsibilities with matched capacity",
        "cells_evaluated": len(cells),
        "promotable_cells": promotable,
        "promotable_count": len(promotable),
        "skipped": skipped,
        "seeds": list(seeds),
        "cell_results": cell_results,
        "promotion_rule": (
            "all 7 seed point deltas below zero and patient, care-unit, and "
            "forward-time 95% CI below zero versus both base and placebo"
        ),
        "mimic_labels_used_for_fit_or_selection": False,
        "runtime_changed": False,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    (args.output_dir / "subtype_external_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "# Physiological Subtype External Validation",
        "",
        "- Discovery: eICU only",
        "- External lockbox: MIMIC-IV only",
        "- Comparator: current release base point and patient-shuffled equal-capacity placebo",
        f"- Evaluated cells: {len(cells)}",
        f"- Promotable cells: {len(promotable)}",
        f"- Promotable: {', '.join(promotable) if promotable else 'none'}",
        "- Runtime changed: no",
        "",
        "| Cell | Base MAE | Subtype MAE | Placebo MAE | External pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for item in cell_results:
        lines.append(
            f"| {item['cell']} | {item['base_mae']:.5g} | "
            f"{item['subtype_mae']:.5g} | {item['placebo_mae']:.5g} | "
            f"{'yes' if item['external_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "MIMIC labels were not used to fit the router, subtype priors, hyperparameters, or cell selection.",
            "Only an exact cell that passes every listed gate is eligible for later runtime integration.",
        ]
    )
    (args.output_dir / "SUBTYPE_EXTERNAL_VALIDATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="7,19,31,43,59,71,83")
    parser.add_argument("--subtypes", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--min-effective-subjects", type=float, default=30.0)
    parser.add_argument("--subtype-prior-strength", type=float, default=20.0)
    parser.add_argument("--subtype-rows-per-subject", type=int, default=4)
    parser.add_argument("--max-rows-per-subject", type=int, default=6)
    parser.add_argument("--discovery-rows-per-cell", type=int, default=2000)
    parser.add_argument("--external-rows-per-cell", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "cells_evaluated": result["cells_evaluated"],
                "promotable_count": result["promotable_count"],
                "promotable_cells": result["promotable_cells"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""The online marginal M for one ICU stay at one hospital, in its v13 lockbox form (``m_weights.py``).

Per MIMIC training sample (five, each with its own hospital state):

1. expert anchors of the 36 cells at the stay's latest anchor, from histories over the stay's 3-hour anchor stream
   (as ``build_expert_anchor_columns`` computed them on the JEPA event examples);
2. the own-trajectory residual point (``stacked_expert_dev.anchor_features``: current value, measurement age, the
   anchors' forecast changes, and trajectories over earlier anchors of the cohort, i.e. anchors with >= 2 organ
   modules observed);
3. the hospital's online point correction and Mondrian MultiQT (``osler_api.online_state``), interpolated at the 50
   member levels; single-event probabilities are member shares;
4. on request, the Gaussian copula couples the members of each block vector (ensemble copula coupling).

Output policy, from the registry (``adaptive_replication_registry_20261102.json``):
- nothing is output for a warm-up stay (the hospital's first ``WARMUP_STAYS`` stays), though its forecasts still
  update the state;
- an event probability is output only where its single_event_probability entry is a candidate;
- every layer's status is reported: own-trajectory points, single events and both copula blocks are candidates from
  one external lockbox (INSPIRE), not independently replicated.
Research runtime only.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from adaptive_cell_weights import predict
from joint_forecast_dev import EVENTS
from mimic_to_eicu_49_cell_external_audit import stable_seed
from online_joint_dev import MEMBERS, at_member_levels, frozen_multiqt
from osler_api import online_state
from osler_api.runtime import REGISTRY, ROOT, stay_frame
from osler_jepa.deployment_clock import WARMUP_STAYS
from osler_jepa.event_ledger import DEFAULT_TARGET_LABEL_POLICIES, TargetLabelPolicy
from osler_jepa.frozen_protocol import sha256
from osler_jepa.joint_copula import couple, latent_draws
from site_adaptive_external_audit import frozen_point_config
from stacked_expert_dev import anchor_features

WEIGHTS = ROOT / "m_weights_20261108"
POINT_DEV = ROOT / "site_adaptive_point_amendment_20261024.json"
MULTIQT = ROOT / "threshold_event_candidates_20261029.json"
BLOCKS = ("cross_target", "across_horizons")
BOUNDARY = {
    "research_only": True,
    "causal_claim_allowed": False,
    "clinical_promotion_allowed": False,
    "form": "v13 lockbox form of M: expert anchor -> own-trajectory residual -> per-hospital online point correction "
            "and Mondrian MultiQT -> Gaussian copula (on request)",
    "evidence": "own-trajectory points, single events and both copula blocks are candidates from one external "
                "lockbox (INSPIRE); none is independently replicated",
    "combined_value": "point and probability are medians over the five MIMIC training samples, each with its own "
                      "hospital state; the median itself was not scored in any lockbox",
    "known_limits": [
        "across-horizon joint members score worse than the static Mondrian marginal on the variogram score "
        "(development: eICU +4.7%, NWICU +5.9%, Zigong +29%); no claim about trajectory shapes",
        "potassium >= 6.0 and bicarbonate < 18 probabilities are not calibrated and are not output",
        "the cross-target copula gain is not material (development -0.1% to -0.5%, INSPIRE about -0.03%)",
        "joint event probabilities are not output",
    ],
}


@dataclass(frozen=True)
class MModel:
    samples: tuple[dict, ...]
    contract: dict
    point_config: object
    multiqt: object
    statuses: dict[str, dict[str, str]]
    metadata: dict

    @property
    def cells(self) -> list[str]:
        return list(self.samples[0]["copula"]["cells"])


def load_m(weights: Path = WEIGHTS, registry: Path = REGISTRY) -> MModel:
    manifest = json.loads((weights / "MANIFEST.json").read_text(encoding="utf-8"))
    contract = json.loads((weights / "CONTRACT.json").read_text(encoding="utf-8"))
    if sklearn.__version__ != manifest["versions"]["scikit-learn"]:
        raise RuntimeError(f"M weights need scikit-learn {manifest['versions']['scikit-learn']}, found {sklearn.__version__}")
    if contract["examples_sha256"] != manifest["examples_sha256"].get(contract["examples"]):
        raise RuntimeError("the input contract was not read from the examples the weights were fitted on")
    point_config, multiqt = frozen_point_config(POINT_DEV), frozen_multiqt(MULTIQT)
    if repr(point_config) != manifest["point_config"] or repr(multiqt) != manifest["multiqt"]:
        raise RuntimeError("the frozen point-correction or MultiQT configuration differs from the export")
    samples = []
    for entry in manifest["samples"]:
        path = weights / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"{entry['file']} does not match its manifest checksum")
        samples.append(joblib.load(path))
    if [s["sample"] for s in samples] != [0, 1, 2, 3, 4]:
        raise RuntimeError("M needs all five MIMIC training samples")
    statuses: dict[str, dict[str, str]] = {}
    for entry in json.loads(registry.read_text(encoding="utf-8"))["cells"]:
        if entry["task_type"] in ("own_trajectory_point", "single_event_probability", "joint_dependence"):
            statuses.setdefault(entry["task_type"], {})[entry["cell"]] = entry["status"]
    metadata = {"weights": weights.name, "registry": registry.name, "registry_sha256": sha256(registry),
                "training_database": "MIMIC-IV v3.1", "examples": contract["examples"]}
    return MModel(samples=tuple(samples), contract=contract, point_config=point_config[1], multiqt=multiqt,
                  statuses=statuses, metadata=metadata)


def inputs(model: MModel, anchors: list[dict]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The stay's stream in the examples' columns, with recomputed histories and observed flags; also the names it
    ignored (outside the contract) and recomputed."""

    allowed = set(model.contract["state_columns"])
    supplied = {column for anchor in anchors for column in anchor["values"]}
    recomputed = sorted(c for c in supplied if c.startswith(("lag1__", "slope1h__", "_observed_")) or c == "hours_since_onset")
    ignored = sorted(supplied - allowed - set(recomputed))
    kept = [{"time": a["time"], "values": {k: v for k, v in a["values"].items() if k in allowed}} for a in anchors]
    frame, _ = stay_frame(kept, model.contract["anchor_spacing_hours"])
    flags = {f"_observed_{v}": (pd.to_numeric(frame[f"{v}_t"], errors="coerce").notna().astype(np.float64)
                                 if f"{v}_t" in frame else pd.Series(0.0, index=frame.index))
             for v in model.contract["observed_flags"]}
    return pd.concat([frame, pd.DataFrame(flags, index=frame.index)], axis=1), ignored, recomputed


def in_cohort(model: MModel, frame: pd.DataFrame) -> np.ndarray:
    """``stacked_expert_dev.load_cohort``'s row filter: at least ``min_modules`` organ modules with an observed variable."""

    count = np.zeros(len(frame), dtype=np.int16)
    for members in model.contract["modules"].values():
        observed = np.zeros(len(frame), dtype=bool)
        for variable in members:
            observed |= frame.get(f"_observed_{variable}", pd.Series(0.0, index=frame.index)).to_numpy() > 0.5
        count += observed
    return count >= model.contract["min_modules"]


def cell_points(model: MModel, frame: pd.DataFrame, cohort: np.ndarray) -> list[dict]:
    """Per sample and cell at the latest anchor: current value, expert anchor and own-trajectory residual point."""

    first = model.samples[0]
    targets, horizons, lags = tuple(first["targets"]), tuple(first["horizons"]), tuple(first["lags"])
    last = frame.iloc[[-1]].reset_index(drop=True)
    stream = frame[cohort].reset_index(drop=True)
    output = []
    for bundle in model.samples:
        anchors = {cell: float(predict(expert, last)[0]) for cell, expert in bundle["experts"].items()}
        long = pd.concat([stream[["stay_id"] + [c for t in targets for c in (f"{t}_t", f"{t}_age_hr") if c in stream]]
                          .assign(t=stream["anchor_time"], horizon_hours=float(h),
                                  **{f"expert__anchor_{t}": np.nan for t in targets}) for h in horizons], ignore_index=True)
        for t in targets:
            for column in (f"{t}_t", f"{t}_age_hr"):
                if column not in long:
                    long[column] = np.nan
            for h in horizons:
                long.loc[(long["t"] == last["anchor_time"].iloc[0]) & (long["horizon_hours"] == float(h)),
                         f"expert__anchor_{t}"] = anchors[f"{t}@{h}h"]
        features = anchor_features(long, targets, horizons, lags).loc[[("request", last["anchor_time"].iloc[0])]]
        features = features.reset_index(drop=True)
        cells = {}
        for t in targets:
            current = float(pd.to_numeric(last.get(f"{t}_t", pd.Series([np.nan])), errors="coerce").iloc[0])
            for h in horizons:
                cell = f"{t}@{h}h"
                anchor = anchors[cell]
                point = (float(bundle["residuals"][cell].predict(features, np.array([anchor]))[0])
                         if np.isfinite(current) else np.nan)
                cells[cell] = {"current": current, "anchor": anchor, "point": point}
        output.append(cells)
    return output


def issue_cell(model: MModel, sample: int, cell: str, state: dict, key: str, values: dict) -> dict:
    """Issue one cell of one sample from the hospital's state; returns the corrected point and the 50 members."""

    marginal = model.samples[sample]["marginals"][cell]
    current = values["current"]
    group = int(np.digitize(current, marginal["edges"]))
    scale = float(np.maximum(marginal["scale"](np.array([current]), np.array([np.nan]))[0], marginal["floor"]))
    corrected, normalized = online_state.issue(
        state, key=key, current=current, prediction=values["point"], group=group, scale=scale,
        base=np.asarray(marginal["bases"][group]), point_scale=marginal["point_scale"], point_config=model.point_config,
        multiqt=model.multiqt)
    quantiles = corrected + scale * normalized
    members = at_member_levels(quantiles[None, :], np.asarray(model.multiqt.levels))[0]
    return {"corrected": corrected, "quantiles": quantiles, "members": members}


def event_probabilities(model: MModel, cell: str, issued: list[dict], current: float) -> list[dict]:
    target, horizon = cell.split("@")
    output = []
    for label, (event_target, event) in EVENTS.items():
        if event_target != target:
            continue
        name = f"{label}@{horizon}"
        status = model.statuses["single_event_probability"].get(name, "not_in_registry")
        shares = [float(np.mean(event(item["members"], current))) for item in issued]
        allowed = status == "candidate"
        output.append({"event": label, "status": status, "probability": float(np.median(shares)) if allowed else None,
                       "sample_probabilities": shares if allowed else None})
    return output


def joint_members(model: MModel, forecast_id: str, issued: dict[str, list[dict]]) -> dict:
    """Coupled members of every block vector whose cells were issued (Gaussian copula, ensemble copula coupling)."""

    first = model.samples[0]
    targets, horizons = first["targets"], first["horizons"]
    vectors = {"cross_target": [[f"{t}@{h}h" for t in targets] for h in horizons],
               "across_horizons": [[f"{t}@{h}h" for h in horizons] for t in targets]}
    output = {}
    for block in BLOCKS:
        entries = []
        for vector in vectors[block]:
            cells = [c for c in vector if c in issued]
            if len(cells) < 2:
                continue
            per_sample = []
            for sample, bundle in enumerate(model.samples):
                order = bundle["copula"]["cells"]
                position = [order.index(c) for c in cells]
                correlation = np.asarray(bundle["copula"]["correlation"])[np.ix_(position, position)]
                _, latent = latent_draws(1, MEMBERS, correlation,
                                         seed=stable_seed("osler_api_joint", forecast_id, sample, block, *cells))
                sorted_members = np.sort(np.stack([issued[c][sample]["members"] for c in cells], axis=-1)[None], axis=1)
                per_sample.append(couple(sorted_members, latent)[0].tolist())
            entries.append({"cells": cells, "members_by_sample": per_sample})
        output[block] = {"status": model.statuses["joint_dependence"].get(block, "not_in_registry"), "vectors": entries}
    return output


def label_policy(target: str) -> TargetLabelPolicy:
    return DEFAULT_TARGET_LABEL_POLICIES.get(target, TargetLabelPolicy(1.0, 4.0))


def check_outcome(target: str, anchor_time: pd.Timestamp, forecast_time: pd.Timestamp, measured_at: pd.Timestamp) -> None:
    """The training label rule (``EventLedger.future_label``, weight > 0): a measurement after the anchor within the
    target's strict tolerance of the forecast time."""

    policy = label_policy(target)
    if (measured_at - anchor_time) / pd.Timedelta(hours=1) < policy.minimum_delta_hours:
        raise ValueError("the outcome must be measured after the forecast's anchor")
    if abs((measured_at - forecast_time) / pd.Timedelta(hours=1)) > policy.strict_tolerance_hours:
        raise ValueError(f"{target} outcomes must be measured within {policy.strict_tolerance_hours:g} h of the forecast time")


def forecast(model: MModel, store, hospital: str, stay: str, anchors: list[dict], *, joint: bool = False) -> dict:
    """Issue M at the stay's latest anchor from the hospital's state, keep it pending, and return what may be output."""

    frame, ignored, recomputed = inputs(model, anchors)
    cohort = in_cohort(model, frame)
    anchor_time = frame["anchor_time"].iloc[-1]
    forecast_id = uuid.uuid4().hex
    result = {"forecast_id": forecast_id, "hospital_id": hospital, "anchor_time": anchor_time.isoformat(),
              "anchors_received": len(anchors), "ignored_columns": ignored, "recomputed_columns": recomputed,
              "model": model.metadata, "boundary": BOUNDARY}
    if not cohort[-1]:
        return {**result, "status": "outside_cohort", "warm_up": None, "cells": [],
                "detail": f"the latest anchor has fewer than {model.contract['min_modules']} organ modules observed"}
    points = cell_points(model, frame, cohort)
    issued: dict[str, list[dict]] = {}
    with store.transaction() as tx:
        previous = tx.latest_anchor(hospital, stay)
        if previous is not None and pd.Timestamp(previous) >= anchor_time:
            raise ValueError("each anchor of a stay is forecast once, in time order: this stay already has a forecast "
                             f"at {previous}")
        arrival, seen = tx.arrival(hospital, stay)
        stale = tx.expired(hospital)
        for cell in model.cells:
            if not np.isfinite(points[0][cell]["current"]):
                continue
            issued[cell] = []
            for sample in range(len(model.samples)):
                state = tx.cell_state(hospital, sample, cell) or online_state.new_cell_state()
                if stale:
                    online_state.expire(state, stale)
                issued[cell].append(issue_cell(model, sample, cell, state, forecast_id, points[sample][cell]))
                tx.save_state(hospital, sample, cell, state)
        horizons = {cell: (anchor_time + pd.Timedelta(hours=int(cell.split("@")[1].rstrip("h")))).isoformat() for cell in issued}
        tx.add_forecast(forecast_id, hospital, stay, anchor_time.isoformat(), horizons)
    warm = arrival < WARMUP_STAYS
    result["warm_up"] = {"stay_is_warm_up": warm, "stays_seen": seen, "warm_up_stays": WARMUP_STAYS}
    if warm:
        return {**result, "status": "warm_up", "cells": [
            {"cell": cell, "forecast_time": horizons[cell], "status": "warm_up"} for cell in issued]}
    cells = []
    for cell, per_sample in issued.items():
        target, horizon = cell.split("@")
        current = points[0][cell]["current"]
        cells.append({
            "cell": cell, "target": target, "horizon_hours": int(horizon.rstrip("h")), "forecast_time": horizons[cell],
            "current_value": current, "status": "ok",
            "own_trajectory_status": model.statuses["own_trajectory_point"].get(cell, "not_in_registry"),
            "point": float(np.median([item["corrected"] for item in per_sample])),
            "samples": [{"sample": s, "expert_anchor": points[s][cell]["anchor"], "residual_point": points[s][cell]["point"],
                         "corrected_point": item["corrected"],
                         "quantiles": dict(zip([str(level) for level in model.multiqt.levels], item["quantiles"].tolist()))}
                        for s, item in enumerate(per_sample)],
            "events": event_probabilities(model, cell, per_sample, current)})
    result = {**result, "status": "ok", "cells": cells}
    if joint:
        result["joint"] = joint_members(model, forecast_id, issued)
    return result


def post_outcome(model: MModel, store, forecast_id: str, cell: str, value: float, measured_at: pd.Timestamp) -> dict:
    """Record a measured outcome of one issued cell; it updates the hospital's state at the cell's next forecast."""

    with store.transaction() as tx:
        record = tx.forecast(forecast_id)
        if record is None:
            raise LookupError("unknown or expired forecast")
        if cell not in record["issued"]:
            raise ValueError(f"{cell} was not issued in this forecast")
        if cell in record["outcomes"]:
            raise ValueError(f"an outcome for {cell} was already recorded")
        check_outcome(cell.split("@")[0], pd.Timestamp(record["anchor_time"]), pd.Timestamp(record["issued"][cell]),
                      measured_at)
        for sample in range(len(model.samples)):
            state = tx.cell_state(record["hospital"], sample, cell)
            if state is None or not online_state.post_outcome(state, forecast_id, value):
                raise RuntimeError(f"{cell} sample {sample}: no pending forecast to update")
            tx.save_state(record["hospital"], sample, cell, state)
        tx.record_outcome(forecast_id, {**record["outcomes"], cell: {"value": value, "measured_at": measured_at.isoformat()}})
    return {"forecast_id": forecast_id, "cell": cell, "status": "recorded",
            "applies": "at this hospital's next forecast of the cell"}

"""Research-only JEPA adapter with measurement-pure history and target heads.

This candidate keeps the v5 checkpoint fixed and learns target-specific factual
residual heads from real MIMIC-IV rows. Its input contains:

* measurement-pure v5 rollout features;
* strictly pre-anchor physiologic level, age, delta, slope, and count;
* observed treatment context and timing.

The default candidate is pure JEPA/history: hand-built belief features are
excluded. ``--include-beliefs`` exists only to reproduce the earlier ablation
and must not be interpreted as the promoted architecture.

The adapter is cross-fitted by stay through ``real_world_improvement``. It is
not a new causal model and cannot promote a JEPA checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dka_world_model import (
    A_DIM,
    ACTION_KEYS,
    S_MEAN,
    S_STD,
    STATE_KEYS,
    a2vec,
    load_checkpoint,
    observation_context,
    s2vec,
    treatment_history_features,
)
from real_world_improvement import (
    TARGET_COLUMNS,
    _action_data,
    _actual_directions,
    _state_from_row,
    episode_features,
    run_crossfit,
    summarize_ablation,
    target_vector,
)
from train_intervention_jepa import choose_device


HISTORY_TARGETS = tuple(
    STATE_KEYS[index] for index in TARGET_COLUMNS if index < len(STATE_KEYS)
)


def _number(value, default=np.nan):
    if value is None or pd.isna(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_physiology_history(frame: pd.DataFrame) -> np.ndarray:
    """Build strictly prior per-target trajectory features.

    Each target contributes [previous value, age, delta, slope, count]. The
    current anchor is added only after its features are emitted, so the block
    cannot use future information or the current value as its own history.
    """

    output = np.zeros((len(frame), len(HISTORY_TARGETS) * 5), dtype=np.float32)
    columns = [f"{name}_t" for name in HISTORY_TARGETS]
    for _, group in frame.groupby("stay_id", sort=False):
        positions = group.sort_values("t").index.to_numpy(dtype=np.int64)
        last = {}
        previous = {}
        for position in positions:
            row = frame.loc[position]
            current_time = _number(row.get("t"), 0.0)
            for target_index, (name, column) in enumerate(zip(HISTORY_TARGETS, columns)):
                current = _number(row.get(column))
                if name in last:
                    prior_value, prior_time, count = last[name]
                    age = max(0.0, current_time - prior_time)
                    delta = prior_value - previous[name][0] if name in previous else 0.0
                    interval = max(prior_time - previous[name][1], 0.25) if name in previous else 1.0
                    slope = delta / interval if name in previous else 0.0
                    output[position, target_index * 5:target_index * 5 + 5] = (
                        prior_value, age, delta, slope, count
                    )
                if np.isfinite(current):
                    if name in last:
                        previous[name] = last[name][:2]
                    last[name] = (current, current_time, last.get(name, (None, None, 0))[2] + 1.0)
    return output


def _observed_state(row, state):
    """Compile only direct anchor observations for JEPA inference."""

    reported = {
        "G": _number(row.get("glucose_t")),
        "HCO3": _number(row.get("bicarbonate_t")),
        "Ke": _number(row.get("potassium_t")),
        "MAP": _number(row.get("map_t")),
    }
    mapping = {
        "pH": "ph_t",
        "anion_gap": "anion_gap_t",
        "Na": "sodium_t",
        "osmolality": "osmolality_t",
        "creatinine": "creatinine_t",
        "urine_output": "urine_output_t",
        "BHB": "BHB_t",
    }
    for key, column in mapping.items():
        value = _number(row.get(column))
        if np.isfinite(value):
            reported[key] = value
    ages = {
        key: _number(row.get(f"{column[:-2]}_age_hr"), 24.0)
        for key, column in mapping.items()
        if f"{column[:-2]}_age_hr" in row.index
    }
    for key, column in (
        ("G", "glucose_t"), ("HCO3", "bicarbonate_t"),
        ("Ke", "potassium_t"), ("MAP", "map_t"),
    ):
        ages[key] = _number(row.get(f"{column[:-2]}_age_hr"), 24.0)
    vector, mask, age = observation_context(reported, ages)
    return vector * mask, mask, age


@torch.no_grad()
def build_examples(
    model,
    frame: pd.DataFrame,
    device,
    *,
    include_beliefs: bool = False,
):
    frame = frame.reset_index(drop=True).copy()
    physiology_history = build_physiology_history(frame)
    if include_beliefs:
        from sequence_worldmodel_vs_ridge import compute_beliefs

        belief_frame = compute_beliefs(frame)
        belief_values = belief_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
    else:
        belief_values = np.zeros((len(frame), 0), dtype=np.float32)
    examples = []
    for row_index, row in frame.iterrows():
        state = _state_from_row(row)
        sequence, treatment_history = _action_data(row)
        if state is None or sequence is None:
            continue
        pure_state, observation_mask, observation_age = _observed_state(row, state)
        treatment_history = (
            np.zeros(A_DIM * 2, dtype=np.float32)
            if treatment_history is None else np.asarray(treatment_history, dtype=np.float32)
        )
        state_tensor = torch.as_tensor(pure_state, dtype=torch.float32, device=device).unsqueeze(0)
        history_tensor = torch.as_tensor(
            treatment_history, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action_tensor = torch.as_tensor(
            sequence, dtype=torch.float32, device=device
        ).unsqueeze(0)
        predicted, _, _ = model.rollout(
            state_tensor,
            action_tensor,
            history_tensor,
            observation_mask=torch.as_tensor(
                observation_mask, dtype=torch.float32, device=device
            ).unsqueeze(0),
            observation_age=torch.as_tensor(
                observation_age, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
        base_normalized = predicted[0, -1].cpu().numpy()
        base_physical = base_normalized * S_STD + S_MEAN
        action_exposure, episode_vector = episode_features(sequence)
        ages = np.asarray([
            min(_number(row.get(f"{name}_age_hr"), 24.0), 24.0) / 24.0
            for name in HISTORY_TARGETS
        ], dtype=np.float32)
        active = bool(row.get("dka_active_t", False))
        onset = min(max(_number(row.get("hours_since_onset"), 0.0), 0.0), 72.0) / 72.0
        shared_parts = [
            pure_state,
            base_normalized - pure_state,
            action_exposure,
            treatment_history,
            episode_vector,
            ages,
            np.asarray([float(active), onset], dtype=np.float32),
            physiology_history[row_index],
        ]
        if include_beliefs:
            shared_parts.append(belief_values[row_index])
        shared = np.concatenate(shared_parts).astype(np.float32)
        propensity = np.concatenate([
            pure_state, treatment_history, ages,
            np.asarray([float(active), onset], dtype=np.float32),
        ]).astype(np.float32)
        latent = model.encode_state(
            state_tensor, history_tensor,
            observation_mask=torch.as_tensor(
                observation_mask, dtype=torch.float32, device=device
            ).unsqueeze(0),
            observation_age=torch.as_tensor(
                observation_age, dtype=torch.float32, device=device
            ).unsqueeze(0),
        )
        symbolic = model.symbolic_outputs(
            latent,
            torch.as_tensor(action_exposure, dtype=torch.float32, device=device).unsqueeze(0),
            delta_hours=sequence.shape[0] * 0.5,
        )
        examples.append({
            "row_index": row_index,
            "stay_id": int(row["stay_id"]),
            "active_dka": active,
            "current_physical": np.asarray([state[name] for name in STATE_KEYS], dtype=np.float32),
            "base_physical": base_physical.astype(np.float32),
            "target_physical": target_vector(row),
            "adapter_features": shared,
            "propensity_features": propensity,
            "action_active": (action_exposure > 0.05).astype(np.int64),
            "episode_weight": float(
                0.25 + 0.75 * episode_vector[1] / (1.0 + episode_vector[2])
            ),
            "actual_directions": _actual_directions(row, state),
            "symbolic_probabilities": symbolic["proposal_logits"].softmax(dim=-1)[0].cpu().numpy(),
        })
    return examples


def summarize_rows(rows):
    output = {}
    for state in STATE_KEYS:
        state_rows = [row for row in rows if row["state"] == state]
        if not state_rows:
            continue
        output[state] = {}
        for method in ("persistence", "base_jepa", "safe_hybrid", "active_safe_hybrid", "ensemble_adapter"):
            values = [abs(row[method] - row["truth"]) for row in state_rows]
            output[state][method] = {"n": len(values), "mae": round(float(np.mean(values)), 4)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--output", default="jepa_belief_adapter_candidate.json")
    parser.add_argument("--predictions", default="jepa_belief_adapter_candidate_predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-beliefs",
        action="store_true",
        help="reproduce the earlier belief-adapter ablation; disabled by default",
    )
    args = parser.parse_args()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    frame = pd.read_parquet(args.mimic)
    examples = build_examples(
        model,
        frame,
        device,
        include_beliefs=args.include_beliefs,
    )
    result = run_crossfit(examples, seed=17)
    predictions = pd.DataFrame(result["numerical_rows"])
    predictions.to_csv(args.predictions, index=False)
    report = {
        "test_type": "measurement-pure JEPA plus physiology-history and target-specific heads",
        "belief_features_used": bool(args.include_beliefs),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cohort": str(Path(args.mimic).resolve()),
        "checkpoint_model": payload.get("metadata", {}).get("model"),
        "examples": len(examples),
        "stays": len({item["stay_id"] for item in examples}),
        "numerical_rows": len(result["numerical_rows"]),
        "all_window": summarize_rows(result["numerical_rows"]),
        "active_dka": summarize_rows([
            row for row in result["numerical_rows"] if row["active_dka"]
        ]),
        "crossfit": {
            "outer_split": "GroupKFold by stay",
            "inner_selection": True,
            "patient_leakage_allowed": False,
            "candidate_only": True,
        },
        "interpretation_boundary": {
            "factual_only": True,
            "causal_claims_allowed": False,
            "checkpoint_promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()

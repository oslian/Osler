"""Train and audit the nonlinear latent rollout candidate.

This is an experiment runner, not a production promotion script.  It combines
the available eICU horizon cohorts for one body-system module, trains a
history-aware latent transition model, and compares it with persistence and a
horizon-specific Ridge baseline.  Patient splits are always by subject.  A
hospital-held-out fit is also run when the cohort has enough hospitals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from nonlinear_latent_rollout import (
    DEFAULT_HORIZONS,
    NonlinearLatentRolloutNet,
    StateScaler,
    effective_rank,
    load_multihorizon_cohort,
    set_seed,
)


def _split_groups(frame: pd.DataFrame, group_column: str, seed: int, fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    groups = frame[group_column].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * fraction)))
    test_groups = set(groups[:n_test])
    test = frame[group_column].isin(test_groups).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _cap_training_rows(frame: pd.DataFrame, rows: np.ndarray, cap_per_horizon: int | None, seed: int) -> np.ndarray:
    if not cap_per_horizon:
        return rows
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    horizon_values = frame["_source_horizon"].to_numpy()
    for horizon in sorted(set(horizon_values[rows].tolist())):
        candidates = rows[horizon_values[rows] == horizon]
        if len(candidates) > cap_per_horizon:
            candidates = rng.choice(candidates, cap_per_horizon, replace=False)
        selected.append(np.asarray(candidates, dtype=np.int64))
    return np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)


def _context_column(frame: pd.DataFrame, train_rows: np.ndarray) -> tuple[np.ndarray, float, float]:
    if "hours_since_onset" not in frame.columns:
        return np.zeros((len(frame), 1), dtype=np.float32), 0.0, 1.0
    raw = pd.to_numeric(frame["hours_since_onset"], errors="coerce").to_numpy(dtype=np.float32)
    train = raw[train_rows]
    median = float(np.nanmedian(train)) if np.isfinite(train).any() else 0.0
    values = np.nan_to_num(raw, nan=median, posinf=median, neginf=median)
    scale = float(np.nanstd(train)) if np.isfinite(train).any() else 1.0
    scale = max(scale, 1.0)
    return ((values - median) / scale).reshape(-1, 1).astype(np.float32), median, scale


def _prepared_arrays(
    frame: pd.DataFrame,
    variables: Sequence[str],
    train_rows: np.ndarray,
) -> tuple[StateScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StateScaler.fit(frame, variables, train_rows)
    current, mask, ages = scaler.normalize_current(frame)
    future, future_mask = scaler.normalize_future(frame)
    context, _, _ = _context_column(frame, train_rows)
    # Values + observation mask + measurement age + disease-time context.
    input_matrix = np.concatenate([current, mask, ages, context], axis=1).astype(np.float32)
    return scaler, input_matrix, current, future, future_mask, mask


def _history_matrix(frame: pd.DataFrame, history: int) -> np.ndarray:
    columns = [f"_history_{i}" for i in range(history)]
    if not all(column in frame.columns for column in columns):
        raise ValueError(f"Missing history columns: {columns}")
    return frame[columns].to_numpy(dtype=np.int64)


def _horizon_tensor(frame: pd.DataFrame, rows: np.ndarray) -> torch.Tensor:
    values = frame["_source_horizon"].to_numpy(dtype=np.float32)[rows]
    return torch.from_numpy(values)


def _predict(
    model: NonlinearLatentRolloutNet,
    input_matrix: np.ndarray,
    current: np.ndarray,
    history_indices: np.ndarray,
    horizons: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outputs: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            sequence = torch.from_numpy(input_matrix[history_indices[batch_rows]]).float()
            state = torch.from_numpy(current[batch_rows]).float()
            horizon = torch.from_numpy(horizons[batch_rows]).float()
            prediction, latent = model(sequence, state, horizon)
            outputs.append(prediction.cpu().numpy())
            embeddings.append(latent.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(embeddings)


def _train_model(
    frame: pd.DataFrame,
    variables: Sequence[str],
    train_rows: np.ndarray,
    *,
    history: int,
    hidden: int,
    latent: int,
    epochs: int,
    batch_size: int,
    lr: float,
    cap_per_horizon: int | None,
    seed: int,
    regimes: int = 3,
    dropout: float = 0.10,
) -> tuple[NonlinearLatentRolloutNet, StateScaler, dict[str, object]]:
    set_seed(seed)
    fit_rows = _cap_training_rows(frame, train_rows, cap_per_horizon, seed)
    scaler, input_matrix, current, future, future_mask, _ = _prepared_arrays(frame, variables, fit_rows)
    history_indices = _history_matrix(frame, history)
    model = NonlinearLatentRolloutNet(
        n_state=len(variables),
        input_dim=input_matrix.shape[1],
        hidden=hidden,
        latent=latent,
        n_regimes=regimes,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    horizon_values = frame["_source_horizon"].to_numpy(dtype=np.float32)
    generator = np.random.default_rng(seed)
    losses: list[float] = []
    regime_usage_history: list[np.ndarray] = []
    regime_entropy_history: list[float] = []
    model.train()
    for epoch in range(epochs):
        order = generator.permutation(fit_rows)
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_rows = order[start:start + batch_size]
            if not len(batch_rows):
                continue
            observations = torch.from_numpy(input_matrix[history_indices[batch_rows]]).float()
            initial_state = torch.from_numpy(current[batch_rows]).float()
            horizon = torch.from_numpy(horizon_values[batch_rows]).float()
            target = torch.from_numpy(future[batch_rows]).float()
            target_mask = torch.from_numpy(future_mask[batch_rows]).float()
            prediction, _, diagnostics = model(
                observations, initial_state, horizon, return_diagnostics=True
            )
            per_value = torch.nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=0.5
            )
            loss = (per_value * target_mask).sum() / target_mask.sum().clamp(min=1.0)
            # Small stability regularizer: discourage a candidate from moving
            # every physiological channel on every rollout step.
            movement = (prediction - initial_state).abs().mean()
            entropy = diagnostics["regime_entropy"]
            regime_usage_history.append(
                diagnostics["regime_usage"].detach().cpu().numpy()
            )
            regime_entropy_history.append(float(entropy.detach().mean().cpu()))
            minimum_entropy = 0.50 * np.log(float(model.n_regimes))
            collapse_penalty = torch.relu(
                torch.as_tensor(minimum_entropy, dtype=entropy.dtype, device=entropy.device)
                - entropy
            ).mean()
            loss = loss + 0.002 * movement + 0.005 * collapse_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(mean_loss)
        print(f"  epoch {epoch + 1}/{epochs} loss={mean_loss:.5f} rows={len(fit_rows)}")
    return model, scaler, {
        "fit_rows": int(len(fit_rows)),
        "loss_history": losses,
        "input_dim": int(input_matrix.shape[1]),
        "history": int(history),
        "regimes": int(model.n_regimes),
        "dropout": float(dropout),
        "regime_usage": (
            np.mean(regime_usage_history, axis=0).tolist()
            if regime_usage_history else []
        ),
        "regime_entropy": (
            float(np.mean(regime_entropy_history))
            if regime_entropy_history else None
        ),
    }


def _ridge_predictions(
    frame: pd.DataFrame,
    variables: Sequence[str],
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    input_matrix: np.ndarray,
    scaler: StateScaler,
) -> np.ndarray:
    """Fit the current linear residual baseline separately per horizon."""

    output = scaler.medians[None, :] + np.zeros((len(test_rows), len(variables)), dtype=np.float32)
    horizons = frame["_source_horizon"].to_numpy(dtype=np.int64)
    future_raw = frame[[f"future_{v}" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    current_raw = frame[[f"{v}_t" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    for target_index, variable in enumerate(variables):
        for horizon in sorted(set(horizons[test_rows].tolist())):
            train_h = train_rows[horizons[train_rows] == horizon]
            test_h = test_rows[horizons[test_rows] == horizon]
            y = future_raw[train_h, target_index]
            usable = np.isfinite(y)
            if usable.sum() < 30:
                output[horizons[test_rows] == horizon, target_index] = current_raw[
                    test_h, target_index
                ]
                continue
            model = Ridge(alpha=10.0)
            model.fit(input_matrix[train_h][usable], y[usable])
            output[horizons[test_rows] == horizon, target_index] = model.predict(
                input_matrix[test_h]
            )
    return output


def _metrics(
    frame: pd.DataFrame,
    variables: Sequence[str],
    rows: np.ndarray,
    candidate_raw: np.ndarray,
    ridge_raw: np.ndarray,
    scaler: StateScaler,
) -> dict[str, object]:
    truth = frame[[f"future_{v}" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)[rows]
    current = frame[[f"{v}_t" for v in variables]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)[rows]
    horizon = frame["_source_horizon"].to_numpy(dtype=np.int64)[rows]
    report: dict[str, object] = {"by_horizon": {}, "by_target": {}}
    for h in sorted(set(horizon.tolist())):
        hmask = horizon == h
        target_report: dict[str, object] = {}
        for index, variable in enumerate(variables):
            valid = hmask & np.isfinite(truth[:, index]) & np.isfinite(current[:, index])
            if valid.sum() < 20:
                continue
            candidate_error = np.abs(candidate_raw[valid, index] - truth[valid, index])
            ridge_error = np.abs(ridge_raw[valid, index] - truth[valid, index])
            persistence_error = np.abs(current[valid, index] - truth[valid, index])
            target_report[variable] = {
                "n": int(valid.sum()),
                "candidate_mae": float(candidate_error.mean()),
                "ridge_mae": float(ridge_error.mean()),
                "persistence_mae": float(persistence_error.mean()),
                "candidate_delta_vs_persistence": float(candidate_error.mean() - persistence_error.mean()),
                "candidate_delta_vs_ridge": float(candidate_error.mean() - ridge_error.mean()),
            }
            report["by_target"].setdefault(variable, []).append({"horizon": int(h), **target_report[variable]})
        report["by_horizon"][str(h)] = target_report
    return report


def run(
    data_root: str | Path,
    module: str,
    horizons: Sequence[int],
    *,
    seeds: int = 3,
    history: int = 5,
    hidden: int = 64,
    latent: int = 48,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_stays_per_horizon: int | None = 8000,
    cap_train_rows_per_horizon: int | None = 40000,
    output: str | Path = "/private/tmp/nonlinear_latent_rollout_candidate.json",
) -> dict[str, object]:
    frame, variables = load_multihorizon_cohort(
        data_root,
        module,
        horizons,
        max_stays_per_horizon=max_stays_per_horizon,
        history=history,
        seed=0,
    )
    print(
        f"rows={len(frame)} subjects={frame['subject_id'].nunique()} "
        f"stays={frame['stay_id'].nunique()} variables={len(variables)} "
        f"horizons={sorted(frame['_source_horizon'].unique().tolist())}"
    )
    result: dict[str, object] = {
        "candidate": True,
        "module": module,
        "horizons": list(horizons),
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "architecture": {
            "history_encoder": "GRU",
            "latent_transition": "bounded nonlinear residual state transition",
            "rollout_step_hours": 6,
            "observation_inputs": ["value", "mask", "measurement_age", "hours_since_onset"],
            "causal_claim_allowed": False,
            "production_promotion_allowed": False,
        },
        "patient_heldout": {},
        "hospital_heldout": {},
    }
    for seed in range(seeds):
        train_rows, test_rows = _split_groups(frame, "subject_id", seed)
        print(f"\npatient seed={seed} train={len(train_rows)} test={len(test_rows)}")
        model, scaler, train_meta = _train_model(
            frame,
            variables,
            train_rows,
            history=history,
            hidden=hidden,
            latent=latent,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cap_per_horizon=cap_train_rows_per_horizon,
            seed=seed,
        )
        _, input_matrix, current, _, _, _ = _prepared_arrays(frame, variables, train_rows)
        history_indices = _history_matrix(frame, history)
        horizon_values = frame["_source_horizon"].to_numpy(dtype=np.float32)
        pred_norm, embeddings = _predict(
            model, input_matrix, current, history_indices, horizon_values, test_rows, batch_size
        )
        candidate_raw = pred_norm * scaler.scales[None, :] + scaler.medians[None, :]
        ridge_raw = _ridge_predictions(frame, variables, train_rows, test_rows, input_matrix, scaler)[
            : len(test_rows)
        ]
        metrics = _metrics(frame, variables, test_rows, candidate_raw, ridge_raw, scaler)
        result["patient_heldout"][str(seed)] = {
            "train": train_meta,
            "metrics": metrics,
            "effective_rank": effective_rank(embeddings),
            "latent_width": int(latent),
        }

    if frame["hospitalid"].nunique() >= 4:
        train_rows, test_rows = _split_groups(frame, "hospitalid", 991)
        print(f"\nhospital heldout train={len(train_rows)} test={len(test_rows)}")
        model, scaler, train_meta = _train_model(
            frame,
            variables,
            train_rows,
            history=history,
            hidden=hidden,
            latent=latent,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cap_per_horizon=cap_train_rows_per_horizon,
            seed=991,
        )
        _, input_matrix, current, _, _, _ = _prepared_arrays(frame, variables, train_rows)
        history_indices = _history_matrix(frame, history)
        horizon_values = frame["_source_horizon"].to_numpy(dtype=np.float32)
        pred_norm, embeddings = _predict(
            model, input_matrix, current, history_indices, horizon_values, test_rows, batch_size
        )
        candidate_raw = pred_norm * scaler.scales[None, :] + scaler.medians[None, :]
        ridge_raw = _ridge_predictions(frame, variables, train_rows, test_rows, input_matrix, scaler)[: len(test_rows)]
        result["hospital_heldout"] = {
            "train": train_meta,
            "metrics": _metrics(frame, variables, test_rows, candidate_raw, ridge_raw, scaler),
            "effective_rank": effective_rank(embeddings),
            "latent_width": int(latent),
        }

    result["promotion_gate"] = {
        "status": "candidate_only",
        "requires": [
            "7 patient-heldout seeds",
            "hospital-heldout improvement",
            "per-target/per-horizon bootstrap confidence intervals",
            "90% interval calibration",
            "non-collapse effective rank",
            "bounded long-rollout stability",
        ],
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--module", default="aki")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--latent", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-stays-per-horizon", type=int, default=8000)
    parser.add_argument("--cap-train-rows-per-horizon", type=int, default=40000)
    parser.add_argument("--output", default="/private/tmp/nonlinear_latent_rollout_candidate.json")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    run(
        args.data_root,
        args.module,
        horizons,
        seeds=args.seeds,
        history=args.history,
        hidden=args.hidden,
        latent=args.latent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_stays_per_horizon=args.max_stays_per_horizon,
        cap_train_rows_per_horizon=args.cap_train_rows_per_horizon,
        output=args.output,
    )


if __name__ == "__main__":
    main()

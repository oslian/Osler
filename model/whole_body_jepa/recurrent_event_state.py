"""Patient-scoped recurrent state for anchor-known measurement events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from whole_body_jepa.continual_learning import PatientResidualState


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureStandardizer":
        array = np.asarray(values, dtype=np.float32)
        mean = array.mean(axis=0)
        scale = array.std(axis=0)
        scale[scale < 1e-5] = 1.0
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(values, dtype=np.float32) - self.mean) / self.scale,
            -8.0,
            8.0,
        ).astype(np.float32)


class RecurrentSignedResidual(nn.Module):
    """Update a persistent patient state and decode a bounded signed residual."""

    def __init__(self, input_width: int, *, hidden_width: int = 32, limit: float = 4.0):
        super().__init__()
        self.limit = float(limit)
        self.event_encoder = nn.Sequential(
            nn.Linear(int(input_width), hidden_width),
            nn.LayerNorm(hidden_width),
            nn.SiLU(),
        )
        self.state_update = nn.GRU(hidden_width, hidden_width, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, values: torch.Tensor, *, reset_each_anchor: bool = False) -> torch.Tensor:
        encoded = self.event_encoder(values)
        if reset_each_anchor:
            batch, length, width = encoded.shape
            flat = encoded.reshape(batch * length, 1, width)
            state, _ = self.state_update(flat)
            state = state.reshape(batch, length, width)
        else:
            state, _ = self.state_update(encoded)
        return self.limit * torch.tanh(self.readout(state).squeeze(-1))


def patient_sequences(frame: pd.DataFrame, selected: np.ndarray | None = None) -> list[np.ndarray]:
    """Return stable, chronological row indices without crossing patient identity."""

    allowed = np.ones(len(frame), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    sequences = []
    for _, group in frame.loc[allowed].groupby("subject_id", sort=True):
        ordered = group.sort_values(["t", "_time_hours"], kind="stable").index.to_numpy(int)
        if len(ordered):
            sequences.append(ordered)
    return sequences


def _batches(sequences: list[np.ndarray], *, seed: int, batch_size: int) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    return [
        [sequences[int(index)] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]


def _padded_batch(
    sequences: list[np.ndarray], features: np.ndarray, target: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(map(len, sequences))
    x = np.zeros((len(sequences), length, features.shape[1]), dtype=np.float32)
    y = np.zeros((len(sequences), length), dtype=np.float32)
    mask = np.zeros((len(sequences), length), dtype=np.float32)
    for position, rows in enumerate(sequences):
        stop = len(rows)
        x[position, :stop] = features[rows]
        valid = np.isfinite(target[rows])
        y[position, :stop] = np.where(valid, target[rows], 0.0)
        mask[position, :stop] = valid
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def predict_recurrent(
    model: RecurrentSignedResidual,
    standardizer: FeatureStandardizer,
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    reset_each_anchor: bool = False,
    feedback_scale: float | None = None,
) -> np.ndarray:
    """Run chronological patient replay and return a correction for every row."""

    normalized = standardizer.transform(features)
    output = np.zeros(len(frame), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for rows in patient_sequences(frame):
            if feedback_scale is not None:
                output[rows] = feedback_sequence(model, standardizer, frame.iloc[rows], features[rows], feedback_scale).cpu().numpy()
                continue
            values = torch.from_numpy(normalized[rows][None])
            prediction = model(values, reset_each_anchor=reset_each_anchor)[0].cpu().numpy()
            output[rows] = prediction.astype(np.float32)
    return output


def feedback_sequence(model, standardizer, frame, features, scale):
    """Differentiable state rollout; only resolved, detached own errors feed back."""
    state = PatientResidualState(prior_strength=3.)
    pending = []
    hidden = None
    predictions = []
    for i, (_, row) in enumerate(frame.iterrows()):
        due = [item for item in pending if item[0] <= row.t]
        pending = [item for item in pending if item[0] > row.t]
        latest = {item[0]: item for item in due}
        for _, error in latest.values():
            state.update(error)
        values = features[i].copy()
        values[:3] = [state.correction / scale, np.log1p(state.count), state.uncertainty_scale(scale) / scale]
        x = torch.from_numpy(standardizer.transform(values)[None, None])
        encoded = model.event_encoder(x)
        latent, hidden = model.state_update(encoded, hidden)
        correction = model.limit * torch.tanh(model.readout(latent).reshape(()))
        predictions.append(correction)
        if np.isfinite(row.truth) and pd.notna(row.future_event_time) and row.future_event_time > row.t:
            prediction = float(row.frozen) + float(correction.detach()) * scale
            pending.append((row.future_event_time, float(row.truth) - prediction))
    return torch.stack(predictions)


def _patient_equal_mae(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    errors = pd.Series(
        np.abs(np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float)),
        index=frame.subject_id.astype(str).to_numpy(),
    )
    return float(errors.groupby(level=0).mean().mean())


def fit_recurrent_signed_residual(
    frame: pd.DataFrame,
    features: np.ndarray,
    fit_mask: np.ndarray,
    tune_mask: np.ndarray,
    *,
    truth: np.ndarray,
    frozen: np.ndarray,
    residual_scale: float,
    seed: int,
    reset_each_anchor: bool = False,
    epochs: int = 60,
    own_feedback: bool = False,
) -> tuple[FeatureStandardizer, RecurrentSignedResidual, dict]:
    """Fit on source patients and select the epoch on disjoint source patients."""

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    standardizer = FeatureStandardizer.fit(features[np.asarray(fit_mask, dtype=bool)])
    normalized = standardizer.transform(features)
    target = np.clip(
        (np.asarray(truth, dtype=float) - np.asarray(frozen, dtype=float)) / residual_scale,
        -4.0,
        4.0,
    ).astype(np.float32)
    model = RecurrentSignedResidual(normalized.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    fit_sequences = patient_sequences(frame, fit_mask)
    if len(fit_sequences) < 50:
        raise ValueError("insufficient patients for recurrent event-state fitting")
    tune = np.asarray(tune_mask, dtype=bool) & np.isfinite(truth)
    baseline_mae = _patient_equal_mae(frame.loc[tune], np.asarray(truth)[tune], np.asarray(frozen)[tune])
    best = {"epoch": -1, "mae": baseline_mae, "state": deepcopy(model.state_dict())}
    patience = 0
    for epoch in range(int(epochs)):
        model.train()
        for batch in _batches(fit_sequences, seed=seed + epoch, batch_size=64):
            if own_feedback:
                losses = []
                for rows in batch:
                    valid = np.isfinite(target[rows])
                    if not valid.any():
                        continue
                    prediction = feedback_sequence(model, standardizer, frame.iloc[rows], features[rows], residual_scale)
                    losses.append(nn.functional.smooth_l1_loss(prediction[valid], torch.from_numpy(target[rows][valid])))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
            else:
                x, y, mask = _padded_batch(batch, normalized, target)
                prediction = model(x, reset_each_anchor=reset_each_anchor)
                row_loss = nn.functional.smooth_l1_loss(prediction, y, reduction="none")
                valid_patients = mask.sum(dim=1) > 0
                if not valid_patients.any():
                    continue
                patient_loss = (row_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                loss = patient_loss[valid_patients].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        correction = predict_recurrent(
            model, standardizer, frame, features, reset_each_anchor=reset_each_anchor,
            feedback_scale=residual_scale if own_feedback else None,
        ) * residual_scale
        tune_mae = _patient_equal_mae(
            frame.loc[tune], np.asarray(truth)[tune], np.asarray(frozen)[tune] + correction[tune]
        )
        if tune_mae < best["mae"] - 1e-4:
            best = {"epoch": epoch, "mae": tune_mae, "state": deepcopy(model.state_dict())}
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break
    if best["state"] is None:
        raise RuntimeError("recurrent event-state fitting produced no checkpoint")
    model.load_state_dict(best["state"])
    return standardizer, model, {
        "selected_epoch": int(best["epoch"]),
        "zero_update_tune_mae": float(baseline_mae),
        "tune_patient_equal_mae": float(best["mae"]),
        "fit_patients": int(frame.loc[np.asarray(fit_mask, dtype=bool), "subject_id"].nunique()),
        "tune_patients": int(frame.loc[np.asarray(tune_mask, dtype=bool), "subject_id"].nunique()),
    }

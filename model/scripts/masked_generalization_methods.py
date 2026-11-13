"""Source-only optimization helpers for masked shared-state diagnostics."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from scripts.audit_release_masked_cells import donor_indices
from scripts.train_release_masked_abc import batch_inputs
from whole_body_jepa.multisource_intervals import SEEDS, patient_split, stable_hash


STEPS = (32, 128, 512)
ALPHAS = (0.1, 1., 10., 100., 1000.)


def source_roles(frame):
    """Entire hospitals reserved before fit/tuning; reject patient crossover."""
    if frame["hospitalid"].isna().any():
        raise ValueError("source hospital provenance missing")
    hospital = frame.hospitalid.astype(str)
    held = hospital.map(lambda h: stable_hash("masked_generalization_hospital", h) % 5 == 0).to_numpy()
    membership = pd.DataFrame({"p": frame.subject_id, "held": held})
    if (membership.groupby("p").held.nunique() > 1).any():
        raise ValueError("patient crosses source hospital roles")
    fit, tune = patient_split(frame.subject_id, 73)
    return {"fit": fit & ~held, "tune": tune & ~held, "hospital_test": held}


def group_weights(units, minimum=20):
    """Half ordinary patient weighting, half equal supported care-unit mass."""
    units = np.asarray(units).astype(str)
    counts = pd.Series(units).value_counts()
    supported = sorted(u for u, n in counts.items() if n >= minimum and u.lower() not in {"unknown", "nan", "none", ""})
    codes = np.asarray([supported.index(u) if u in supported else -1 for u in units])
    uniform = np.full(len(units), 1 / len(units))
    balanced = np.asarray([1 / counts[u] if u in supported else 0. for u in units])
    if balanced.sum():
        balanced /= balanced.sum()
        uniform = .5 * uniform + .5 * balanced
    return uniform, codes, supported


def selection_score(truth, prediction, units):
    """Fixed source-tuning objective; no external outcomes or optimizers here."""
    error = np.abs(np.asarray(truth) - np.asarray(prediction))
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("invalid tuning errors")
    units = np.asarray(units).astype(str)
    means = [float(error[units == u].mean()) for u in sorted(set(units))
             if np.sum(units == u) >= 30 and u.lower() not in {"unknown", "nan", "none", ""}]
    mean = float(error.mean())
    worst = max(means) if means else mean
    return {"mae": mean, "worst_supported_unit_mae": worst,
            "selection_score": .5 * mean + .5 * worst}


def dro_loss(errors, codes, log_weights):
    """Exponentiated-gradient group DRO mixed with ordinary L1 risk.

    Small unsupported groups retain the ordinary-risk term. Weights are
    updated only from observed training minibatches, never validation errors.
    """
    if not len(log_weights):
        return errors.mean()
    means, present = [], []
    for i in range(len(log_weights)):
        mask = codes == i
        if bool(mask.any()):
            means.append(errors[mask].mean())
            present.append(i)
    if not present:
        return errors.mean()
    stacked = torch.stack(means)
    with torch.no_grad():
        log_weights[present] += .01 * stacked.detach()
        log_weights -= torch.logsumexp(log_weights, dim=0)
    weights = torch.softmax(log_weights[present], dim=0).detach()
    return .5 * errors.mean() + .5 * (weights * stacked).sum()


def raw_view(arrays, rows, hidden):
    inputs = batch_inputs(arrays, rows, hidden)
    # Identical legal time/history input as JEPA, including its neutral masks.
    # This probe is not an additional deployed body model.
    return torch.cat([inputs[0].flatten(1), inputs[1]], dim=1).numpy()


def cached_context(model, arrays, rows, hidden):
    contexts, pre_contexts = [], []

    def capture(_module, values):
        pre_contexts.append(values[0].detach().cpu().numpy().copy())

    handle = model.state_encoder.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                contexts.append(model.encode_context_components(
                    *batch_inputs(arrays, rows[start:start + 256], hidden))[0].numpy())
    finally:
        handle.remove()
    return torch.from_numpy(np.concatenate(contexts)), torch.from_numpy(np.concatenate(pre_contexts))


def fit_curve(initial, state, truth, units, seed, method, steps=STEPS):
    head = copy.deepcopy(initial).eval()
    for p in head.parameters():
        p.requires_grad_(True)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.001)
    weights, codes, names = group_weights(units)
    codes = torch.as_tensor(codes)
    log_weights = torch.zeros(len(names))
    snapshots = {}
    for step in range(1, max(steps) + 1):
        rows = rng.choice(len(state), 128, replace=True, p=weights if method == "balanced" else None)
        optimizer.zero_grad(set_to_none=True)
        error = (head(state[rows]).flatten() - truth[rows]).abs()
        loss = dro_loss(error, codes[rows], log_weights) if method == "dro" else error.mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if step in steps:
            snapshots[step] = copy.deepcopy(head).eval()
    return snapshots


def ensemble_predict(heads, state):
    with torch.no_grad():
        return np.mean([h(state).flatten().numpy() for h in heads], axis=0)


def fit_neural_method(initial, fit, tune, method, target):
    by_step = {s: [] for s in STEPS}
    placebo_by_step = {s: [] for s in STEPS}
    placebo_tuning = {s: [] for s in STEPS}
    for seed in SEEDS:
        draw = stable_hash("generalization_fit", target, seed)
        permutation = donor_indices(fit["subjects"], stable_hash("fit_placebo", target, seed))
        validation_permutation = donor_indices(tune["subjects"], stable_hash("tune_placebo", target, seed))
        a = fit_curve(initial, fit["latent"], fit["y"], fit["units"], draw, method)
        p = fit_curve(initial, fit["latent"][permutation], fit["y"], fit["units"], draw, method)
        for step in STEPS:
            by_step[step].append(a[step])
            placebo_by_step[step].append(p[step])
            placebo_tuning[step].append(ensemble_predict([p[step]], tune["latent"][validation_permutation]))
    curve, placebo_curve = {}, {}
    for step in STEPS:
        curve[step] = selection_score(tune["y"], ensemble_predict(by_step[step], tune["latent"]), tune["units"])
        curve[step]["train_mae"] = float(np.abs(ensemble_predict(by_step[step], fit["latent"]) - fit["y"].numpy()).mean())
        placebo_curve[step] = selection_score(tune["y"], np.mean(placebo_tuning[step], axis=0), tune["units"])
    chosen = min(STEPS, key=lambda s: (curve[s]["selection_score"], s))
    chosen_placebo = min(STEPS, key=lambda s: (placebo_curve[s]["selection_score"], s))
    return {"heads": by_step[chosen], "placebo_heads": placebo_by_step[chosen_placebo],
            "fixed32_heads": by_step[32], "fixed32_placebo_heads": placebo_by_step[32],
            "chosen_steps": chosen, "placebo_chosen_steps": chosen_placebo,
            "curve": curve, "placebo_curve": placebo_curve}


def fit_ridge_probe(fit, tune, view, target):
    fitted, choices, placebo, placebo_choices = {}, {}, {}, {}
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(fit[view], fit["y"].numpy())
        fitted[alpha] = model
        choices[alpha] = selection_score(tune["y"], model.predict(tune[view]), tune["units"])
        placebo[alpha], preds = [], []
        for seed in SEEDS:
            p = donor_indices(fit["subjects"], stable_hash("fit_placebo", target, seed))
            v = donor_indices(tune["subjects"], stable_hash("tune_placebo", target, seed))
            other = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
            other.fit(np.asarray(fit[view])[p], fit["y"].numpy())
            placebo[alpha].append(other)
            preds.append(other.predict(np.asarray(tune[view])[v]))
        placebo_choices[alpha] = selection_score(tune["y"], np.mean(preds, axis=0), tune["units"])
    chosen = min(ALPHAS, key=lambda a: (choices[a]["selection_score"], -a))
    pc = min(ALPHAS, key=lambda a: (placebo_choices[a]["selection_score"], -a))
    return {"model": fitted[chosen], "placebos": placebo[pc], "alpha": chosen,
            "placebo_alpha": pc, "curve": choices, "placebo_curve": placebo_choices}

"""Robust, source-selected readouts of the existing frozen shared body state."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.audit_release_masked_cells import donor_indices
from scripts.masked_generalization_methods import selection_score
from whole_body_jepa.multisource_intervals import SEEDS, stable_hash


FAMILIES = {
    "latent_median": ({"alpha": .01}, {"alpha": .1}),
    "latent_huber": ({"alpha": .1}, {"alpha": 1.}, {"alpha": 10.}),
    "latent_hist_mae": ({"max_leaf_nodes": 7}, {"max_leaf_nodes": 15}),
}


def estimator(family, parameters, seed):
    if family == "latent_median":
        model = QuantileRegressor(quantile=.5, solver="highs", **parameters)
    elif family == "latent_huber":
        model = HuberRegressor(epsilon=1.35, max_iter=2000, tol=1e-5, **parameters)
    elif family == "latent_hist_mae":
        model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=128, learning_rate=.05,
            min_samples_leaf=50, l2_regularization=10., early_stopping=False,
            random_state=seed, **parameters)
    else:
        raise ValueError(f"unknown readout family: {family}")
    return make_pipeline(StandardScaler(), model)


def resampling(subjects, target, seed):
    subjects = np.asarray(subjects)
    permutation = donor_indices(subjects, stable_hash("robust_fit_placebo", target, seed))
    rows = np.random.default_rng(stable_hash("robust_bag", target, seed)).choice(
        len(subjects), len(subjects), replace=True)
    return rows, permutation


def ensemble(models, features):
    predictions = np.stack([model.predict(features) for model in models])
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite readout predictions")
    return predictions.mean(axis=0)


def placebo_ensemble(models, features, subjects, salt):
    if len(models) != len(SEEDS):
        raise ValueError("expected seven matched placebo models")
    predictions = [model.predict(features[donor_indices(subjects, stable_hash(salt, seed))])
                   for seed, model in zip(SEEDS, models)]
    result = np.mean(predictions, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite placebo predictions")
    return result


def fit_family(family, fit, tune, target):
    x, y = np.asarray(fit["latent"], dtype=float), np.asarray(fit["y"], dtype=float)
    tx, ty = np.asarray(tune["latent"], dtype=float), np.asarray(tune["y"], dtype=float)
    if set(fit["subjects"]) & set(tune["subjects"]):
        raise ValueError("fit/tune patient overlap")
    if not all(np.isfinite(v).all() for v in (x, y, tx, ty)):
        raise ValueError("nonfinite fitting inputs")
    fitted, controls, trace = {}, {}, {}
    for index, parameters in enumerate(FAMILIES[family]):
        real, placebo = [], []
        try:
            for seed in SEEDS:
                rows, permutation = resampling(fit["subjects"], target, seed)
                a, p = estimator(family, parameters, seed), estimator(family, parameters, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    a.fit(x[rows], y[rows])
                    p.fit(x[permutation[rows]], y[rows])
                real.append(a)
                placebo.append(p)
            score = selection_score(ty, ensemble(real, tx), tune["units"])
            control_score = selection_score(
                ty, placebo_ensemble(placebo, tx, tune["subjects"], f"robust_tune:{target}"), tune["units"])
        except (ConvergenceWarning, ValueError) as error:
            trace[index] = {"parameters": parameters, "status": "fit_failed", "reason": str(error)}
            continue
        fitted[index], controls[index] = real, placebo
        trace[index] = {"parameters": parameters, "status": "fitted", "real": score, "placebo": control_score,
                        "train_mae": float(np.abs(ensemble(real, x) - y).mean())}
    if not fitted:
        return None, {"status": "all_configurations_failed", "trace": trace}
    selected = min(fitted, key=lambda i: (trace[i]["real"]["selection_score"], i))
    pc = min(controls, key=lambda i: (trace[i]["placebo"]["selection_score"], i))
    # Keep both an independently selected placebo and one with identical chosen
    # complexity to the real ensemble. Passing only a weaker control is not enough.
    bundle = {"models": fitted[selected], "placebos": controls[pc],
              "matched_placebos": controls[selected]}
    info = {"status": "fitted", "selected": selected, "placebo_selected": pc,
            "selection_score": trace[selected]["real"]["selection_score"], "trace": trace}
    return bundle, info


def value_strata(truth, predictions, fit_bounds):
    """Outcome strata are evaluation only, never routing or input features."""
    y = np.asarray(truth)
    lo, hi = fit_bounds
    masks = {"low": y < lo, "middle": (y >= lo) & (y <= hi), "high": y > hi}
    return {name: {"patients": int(mask.sum()), "models": {
                arm: {"mae": float(np.abs(pred[mask] - y[mask]).mean()),
                      "bias": float(np.mean(pred[mask] - y[mask]))}
                for arm, pred in predictions.items()} if mask.any() else {}}
            for name, mask in masks.items()}

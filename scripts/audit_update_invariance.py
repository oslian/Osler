"""Fixed-weight documentation-density and immediate feedback attribution probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_recurrent_event_state_loop import prepare_replay, patient_equal_mae
from scripts.audit_event_aware_patient_state import patient_partition, paired
from whole_body_jepa.continual_learning import PatientResidualState
from whole_body_jepa.recurrent_event_state import fit_recurrent_signed_residual, predict_recurrent, patient_sequences
from whole_body_jepa.multisource_intervals import SEEDS, sha256


def probe_patient(model, standardizer, frame, features, scale, repeats=1):
    """Duplicate consumption, never duplicate an outcome; branch feedback locally."""
    state = PatientResidualState(prior_strength=3.)
    pending, hidden, outputs, withheld, metadata = [], None, [], [], []

    def step(values, memory, prior):
        x = values.copy()
        x[:3] = [memory.correction / scale, np.log1p(memory.count), memory.uncertainty_scale(scale) / scale]
        encoded = model.event_encoder(torch.from_numpy(standardizer.transform(x)[None, None]))
        latent, next_hidden = model.state_update(encoded, prior)
        correction = model.limit * torch.tanh(model.readout(latent).reshape(()))
        return float(correction) * scale, next_hidden

    for i, (_, row) in enumerate(frame.iterrows()):
        before = deepcopy(state)
        due = {item[0]: item for item in pending if item[0] <= row.t}
        pending = [item for item in pending if item[0] > row.t]
        changed, ages = False, []
        for due_time, error, origin in due.values():
            state.update(error)
            changed |= bool(frame.iloc[origin + 1:i + 1].treatment_changed.any())
            ages.append((row.t - frame.iloc[origin].t).total_seconds() / 3600)
        no_new_error, _ = step(features[i], before, hidden)
        correction, hidden = step(features[i], state, hidden)
        outputs.append(float(row.frozen) + correction)
        withheld.append(float(row.frozen) + no_new_error)
        metadata.append((bool(due), changed, max(ages, default=0.)))
        if np.isfinite(row.truth) and pd.notna(row.future_event_time) and row.future_event_time > row.t:
            pending.append((row.future_event_time, float(row.truth) - outputs[-1], i))
        # Exact same event at the same time. No new forecast, truth or feedback.
        # The current scored prediction remains fixed; only later states change.
        for _ in range(repeats - 1):
            _, hidden = step(features[i], state, hidden)
    return np.asarray(outputs), np.asarray(withheld), np.asarray(metadata)


def probe_stream(model, standardizer, frame, features, scale, repeats=1):
    pred, withheld = np.zeros(len(frame)), np.zeros(len(frame))
    meta = np.zeros((len(frame), 3))
    model.eval()
    with torch.inference_mode():
        for rows in patient_sequences(frame):
            pred[rows], withheld[rows], meta[rows] = probe_patient(model, standardizer, frame.iloc[rows], features[rows], scale, repeats)
    return pred, withheld, meta


def evaluate(model, normalizer, frame, features, scale, seed):
    normal, withheld, meta = probe_stream(model, normalizer, frame, features, scale)
    production = frame.frozen.to_numpy() + scale * predict_recurrent(model, normalizer, frame, features, feedback_scale=scale)
    np.testing.assert_allclose(normal, production, atol=2e-5, rtol=1e-6)
    primary = np.isfinite(frame.truth) & (frame.resolved_history_count > 0)
    result = {"production_replay_max_difference": float(np.max(np.abs(normal-production))), "density": {}, "feedback": {}}
    for repeats in (2, 4):
        duplicated, _, _ = probe_stream(model, normalizer, frame, features, scale, repeats)
        shifts = np.abs(duplicated[primary] - normal[primary])
        result["density"][str(repeats)] = {
            "mean_absolute_prediction_shift": float(shifts.mean()),
            "p95_absolute_prediction_shift": float(np.quantile(shifts, .95)),
            "fraction_shift_over_1_mg_dl": float(np.mean(shifts > 1)),
            "mae_difference_ci": paired(frame.loc[primary], duplicated[primary], normal[primary], f"density:{seed}:{repeats}", 1000),
        }
    masks = {"all_primary": primary, "feedback_arrives": primary & (meta[:, 0] > 0),
             "arrives_after_treatment_snapshot_change": primary & (meta[:, 0] > 0) & (meta[:, 1] > 0),
             "arrives_without_detected_treatment_change": primary & (meta[:, 0] > 0) & (meta[:, 1] == 0)}
    for name, mask in masks.items():
        work = frame.loc[mask]
        entry = {"rows": len(work), "patients": int(work.subject_id.nunique())}
        if mask.any():
            entry.update({"normal_mae": patient_equal_mae(work, work.truth, normal[mask]),
                          "frozen_mae": patient_equal_mae(work, work.truth, work.frozen),
                          "withheld_current_feedback_mae": patient_equal_mae(work, work.truth, withheld[mask]),
                          "withheld_minus_normal": paired(work, withheld[mask], normal[mask], f"feedback:{seed}:{name}", 1000),
                          "median_feedback_origin_age_hours": float(np.median(meta[mask, 2]))})
        result["feedback"][name] = entry
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    root = Path('/Users/chunyouchang/Downloads')
    args.model_dir = Path('models')
    args.eicu = Path('/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.mimic = Path('/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.eicu_root = root / 'eicu-collaborative-research-database-2.0'
    args.mimic_root = root / 'mimic-iv-3.1'
    args.eicu_context_cache = Path('/private/tmp/event_aware_eicu_glucose_6h_context.npz')
    args.mimic_context_cache = Path('/private/tmp/event_aware_mimic_glucose_6h_context.npz')
    args.scored_only = False
    replay, feature, scale, *_ = prepare_replay(args)
    results = []
    for seed in SEEDS:
        frame = replay['development']
        fit, tune = patient_partition(frame.subject_id, seed)
        normalizer, model, audit = fit_recurrent_signed_residual(frame, feature['development'], fit, tune,
            truth=frame.truth.to_numpy(), frozen=frame.frozen.to_numpy(), residual_scale=scale, seed=seed, own_feedback=True)
        roles = {role: evaluate(model, normalizer, replay[role], feature[role], scale, seed) for role in ('eicu_hospital', 'mimic')}
        results.append({'seed': seed, 'fit': audit, 'roles': roles})
        print(json.dumps({'seed': seed, 'selected_epoch': audit['selected_epoch'], 'complete': True}), flush=True)
    report = {'status': 'diagnostic_only', 'runtime_modified': False, 'promotion_allowed': False,
        'causal_claim_allowed': False, 'untouched_lockbox': False, 'cell': 'glucose@6h',
        'protocol': 'UPDATE_INVARIANCE_PROTOCOL_20260910.md',
        'binding': {name: sha256(Path(name)) for name in [__file__, 'whole_body_jepa/recurrent_event_state.py',
            'scripts/audit_recurrent_event_state_loop.py', 'UPDATE_INVARIANCE_PROTOCOL_20260910.md']},
        'residual_scale': scale, 'seeds': results}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

"""Descriptive fixed-hospital decomposition; never select or promote a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256


SEEDS = (7, 19, 31, 43, 59, 71, 83)
TEMP_ROOT = Path('/private/tmp/rate_anchored_potassium_20260911')
BACKUP_ROOT = Path('data/research_recovery/rate_anchored_potassium_20260911')


def decompose(predictions, truth):
    """Equal weight per fit and per patient; variance uses population divisor."""
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if p.ndim != 2 or y.shape != (p.shape[1],) or not p.size:
        raise ValueError('expected fits by identical patients and one truth vector')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite predictions or truth')
    mean = p.mean(axis=0)
    mse = float(np.mean((p-y)**2))
    spread = float(np.mean((p-mean)**2))
    residual = float(np.mean((mean-y)**2))
    if not np.isclose(mse, spread+residual, rtol=1e-12, atol=1e-12):
        raise ValueError('MSE decomposition failed')
    return dict(mean_fit_mse=mse, between_fit_variance=spread,
                mean_prediction_residual_mse=residual,
                between_fit_mse_fraction=spread/mse if mse else None,
                per_fit_mae=np.mean(abs(p-y), axis=1).tolist(),
                per_fit_signed_bias=np.mean(p-y, axis=1).tolist(),
                median_patient_prediction_sd=float(np.median(p.std(axis=0))),
                p90_patient_prediction_range=float(np.quantile(np.ptp(p, axis=0), .9)),
                descriptive_mean_prediction_mae=float(np.mean(abs(mean-y))))


def align_target(frames, target):
    selected = {}
    for seed, frame in frames.items():
        part = frame.loc[frame.target.eq(target)].copy()
        if part.empty:
            continue
        if part.subject_id.duplicated().any():
            raise ValueError('duplicate patient-target record')
        selected[seed] = part.set_index('subject_id').sort_index()
    if not selected:
        raise ValueError('no target records')
    reference = next(iter(selected.values()))
    for part in selected.values():
        if not part.index.equals(reference.index):
            raise ValueError('hospital patient support differs; do not silently intersect')
        if not np.array_equal(part.truth.to_numpy(), reference.truth.to_numpy()):
            raise ValueError('truth changed between fits')
    return selected, reference.truth.to_numpy(dtype=float)


def main():
    out = Path('artifacts/native_reconstruction_fit_stability_20260917.json')
    if out.exists():
        raise ValueError('existing diagnostic evidence')
    summary_path = Path('artifacts/native_reconstruction_seven_split_summary_20260917.json')
    summary = json.loads(summary_path.read_text())
    bindings = {str(summary_path): sha256(summary_path), str(Path(__file__)): sha256(Path(__file__))}
    for filename, digest in summary['bindings'].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f'changed aggregate dependency: {filename}')
    frames = {}
    for seed in SEEDS:
        name = ('native_reconstruction_process_evaluation_20260914.json' if seed == 7 else
                f'native_reconstruction_replication_v2_{seed}_evaluation_20260914.json')
        report_path = Path('artifacts') / name
        report = json.loads(report_path.read_text())
        if not report['complete']:
            raise ValueError('incomplete evaluation')
        saved = [(Path(p), h) for p, h in report['bindings'].items()
                 if Path(p).name == 'heldout_predictions.pt']
        if len(saved) != 1:
            raise ValueError('ambiguous saved prediction binding')
        original, digest = saved[0]
        durable = BACKUP_ROOT / original.relative_to(TEMP_ROOT)
        if sha256(durable) != digest:
            raise ValueError('durable predictions do not match original report')
        bindings[str(durable)] = digest
        bindings[str(report_path)] = sha256(report_path)
        data = torch.load(durable, weights_only=False)
        for role, records in data.items():
            frame = pd.DataFrame(records)
            if frame.duplicated(['subject_id', 'target']).any():
                raise ValueError('duplicate scored pair')
            if set(frame.target) != set(report['roles'][role]):
                raise ValueError('report target inventory mismatch')
            for target, part in frame.groupby('target'):
                expected = report['roles'][role][target]
                if len(part) != expected['patients']:
                    raise ValueError('report patient count mismatch')
                for arm in ('real', 'process', 'population'):
                    value = float(np.mean(abs(part[arm].to_numpy()-part.truth.to_numpy())))
                    if not np.isclose(value, expected['normalized_mae'][arm], rtol=0, atol=1e-7):
                        raise ValueError('saved predictions do not reproduce report MAE')
            if role == 'hospital':
                frames[seed] = frame
    rows = {}
    for target in summary['targets']:
        aligned, truth = align_target(frames, target)
        arrays = {arm: np.stack([f[arm].to_numpy(dtype=float) for f in aligned.values()])
                  for arm in ('real', 'process', 'population')}
        gains = np.mean(abs(arrays['real']-truth)-abs(arrays['process']-truth), axis=1)
        rows[target] = dict(patients=len(truth), seeds=list(aligned),
            missing_seeds=[s for s in SEEDS if s not in aligned],
            complete_seven=len(aligned)==len(SEEDS), supported=len(truth)>=30,
            arms={arm: decompose(p, truth) for arm, p in arrays.items()},
            real_minus_process_mae_by_fit=gains.tolist(),
            numerical_gain_fits=int(np.sum(gains<0)),
            nominal_combined_gain_splits=summary['targets'][target]['nominal_gain_splits'])
    result = dict(complete=True, targets=rows, bindings=bindings,
        report_mae_reproduction=True, patient_truth_alignment=True,
        runtime_modified=False, promotion_allowed=False,
        limitation=('Posthoc descriptive same-hospital analysis, not new independent evidence. '
                    'Fit composition, source medians, initialization, sampling and epoch selection '
                    'vary together; between-fit variance is not pure optimization variance. '
                    'MSE decomposition does not change the original MAE gate. Mean prediction '
                    'is an arithmetic diagnostic, not a selected or promoted ensemble. '
                    'Remaining mean-prediction error is not irreducible biological noise.'))
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for target, item in rows.items():
        r = item['arms']['real']
        print(target, 'n=', item['patients'], 'fits=', len(item['seeds']),
              'between_fit_mse_fraction=', round(r['between_fit_mse_fraction'] or 0, 4),
              'real_process_mean_signs=', item['numerical_gain_fits'], flush=True)


if __name__ == '__main__':
    main()

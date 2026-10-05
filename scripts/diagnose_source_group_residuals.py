"""Source-only local bias/rank diagnostic, not deployable calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def patient_half(patient):
    digest = hashlib.sha256(('source-residual-v1:'+str(patient)).encode()).digest()
    return digest[0] % 2


def crossfit_offset(truth, prediction, halves):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    halves = np.asarray(halves)
    if (truth.ndim != 1 or prediction.shape != truth.shape or halves.shape != truth.shape
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()
            or not np.isin(halves, [0, 1]).all()):
        raise ValueError('Invalid paired values or halves')
    if not len(truth) or len(np.unique(halves)) < 2:
        return None
    output = np.empty_like(prediction)
    for held in (0, 1):
        fitting = halves != held
        shift = np.median(truth[fitting]-prediction[fitting])
        output[~fitting] = prediction[~fitting]+shift
    return output


def arm_statistics(truth, prediction, halves):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    corrected = crossfit_offset(truth, prediction, halves)
    if not len(truth):
        raise ValueError('Empty group')
    y_iqr = float(np.diff(np.quantile(truth, [.25, .75]))[0])
    p_iqr = float(np.diff(np.quantile(prediction, [.25, .75]))[0])
    rho = (float(spearmanr(truth, prediction).statistic)
           if len(np.unique(truth)) > 1 and len(np.unique(prediction)) > 1 else None)
    return {
        'raw_mae': float(np.mean(abs(prediction-truth))),
        'median_signed_error': float(np.median(prediction-truth)),
        'truth_iqr': y_iqr, 'prediction_iqr': p_iqr,
        'prediction_truth_iqr_ratio': p_iqr/y_iqr if y_iqr > 0 else None,
        'spearman': rho,
        'crossfit_local_offset_mae': float(np.mean(abs(corrected-truth))) if corrected is not None else None,
    }


def group_diagnostic(part):
    if (part.patient.duplicated().any() or not part.fit_scale.gt(0).all()
            or not np.isfinite(part[['truth', 'real', 'process', 'population', 'fit_scale']]).all().all()):
        raise ValueError('Invalid group query')
    scale = part.fit_scale.to_numpy()
    if not np.all(scale == scale[0]):
        raise ValueError('Mixed target scales')
    halves = np.asarray([patient_half(p) for p in part.patient])
    truth = part.truth.to_numpy()/scale
    return {
        'patients': len(part), 'half_counts': [int((halves == i).sum()) for i in (0, 1)],
        'arms': {arm: arm_statistics(truth, part[arm].to_numpy()/scale, halves)
                 for arm in ('real', 'process', 'population')},
    }


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    output = Path('artifacts/native_reconstruction_source_group_residuals_20261001.json')
    if output.exists():
        raise ValueError('Existing diagnostic')
    previous_path = Path('artifacts/native_reconstruction_full_continuation_selection_20260930.json')
    previous = json.loads(previous_path.read_text())
    checked = {}
    verify_bound_files(previous['bindings'], checked)
    paths = [Path(__file__), previous_path, Path('scripts/score_monitored_continuation_outer.py')]
    folds = {}
    for fold in range(5):
        proof_path = Path(f'artifacts/native_reconstruction_continuation_selection_fold{fold}_verification_20260930.json')
        proof = json.loads(proof_path.read_text())
        verify_bound_files(proof['bindings'], checked)
        base_path = root / f'coverage_selection_20260926/fold{fold}.parquet'
        fresh_path = root / f'continuation_selection_20260930/fold{fold}.parquet'
        base, fresh = pd.read_parquet(base_path), pd.read_parquet(fresh_path)
        pd.testing.assert_frame_equal(base[['patient', 'target', 'fold']], fresh[['patient', 'target', 'fold']])
        frame = base.copy()
        for arm in ('real', 'process'):
            frame[arm] = fresh[arm+'10'].to_numpy()
        results = {}
        for (target, component), part in frame.groupby(['target', 'component'], sort=True):
            record = group_diagnostic(part)
            reference = next(g for g in previous['folds'][str(fold)][target]['per_component']
                             if g['component'] == str(component))
            record['recurring_mean_harm'] = {}
            for arm in ('real', 'process', 'population'):
                old_key = arm+'10' if arm != 'population' else arm
                if not np.isclose(record['arms'][arm]['raw_mae'], reference['normalized_mae'][old_key], atol=1e-12, rtol=1e-12):
                    raise ValueError('Source component reproduction failed')
            for control in ('process', 'population'):
                c8 = 'process8' if control == 'process' else control
                c10 = 'process10' if control == 'process' else control
                values = reference['normalized_mae']
                record['recurring_mean_harm'][control] = bool(
                    values['real8'] > values[c8] and values['real10'] > values[c10])
            results.setdefault(target, {})[str(component)] = record
        if len(results) != 47:
            raise ValueError('Incomplete target family')
        folds[str(fold)] = results
        paths.extend((proof_path, base_path, fresh_path))
        print('Diagnosed complete source fold', fold, flush=True)
    summary = {}
    for control in ('process', 'population'):
        rows = [(target, row) for fold in folds.values() for target, groups in fold.items()
                for row in groups.values() if row['patients'] >= 30 and row['recurring_mean_harm'][control]]
        valid = [(t, r) for t, r in rows if r['arms']['real']['crossfit_local_offset_mae'] is not None]
        still_harm = [(t, r) for t, r in valid if r['arms']['real']['crossfit_local_offset_mae']
                      > r['arms'][control]['crossfit_local_offset_mae']]
        rhos = [r['arms']['real']['spearman'] for _, r in rows if r['arms']['real']['spearman'] is not None]
        ratios = [r['arms']['real']['prediction_truth_iqr_ratio'] for _, r in rows
                  if r['arms']['real']['prediction_truth_iqr_ratio'] is not None]
        summary[control] = {
            'supported_recurring_harm_group_occurrences': len(rows),
            'targets_with_supported_recurring_harm': sorted({t for t, _ in rows}),
            'crossfit_supported_group_occurrences': len(valid),
            'crossfit_still_harm_group_occurrences': len(still_harm),
            'targets_with_residual_matched_local_harm': sorted({t for t, _ in still_harm}),
            'real_crossfit_improves_over_uncorrected_groups': sum(
                r['arms']['real']['crossfit_local_offset_mae'] < r['arms']['real']['raw_mae'] for _, r in valid),
            'real_spearman_nonpositive_groups': sum(r <= 0 for r in rhos),
            'real_spearman_estimable_groups': len(rhos),
            'median_real_spearman': float(np.median(rhos)) if rhos else None,
            'median_prediction_truth_iqr_ratio': float(np.median(ratios)) if ratios else None,
        }
    output.write_text(json.dumps(dict(summary=summary, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='All47 targets/source only. Local offsets are diagnostic two-way crossfitting within known source components, not deployment or cross-hospital validation. Same patient hash split across targets/folds; overlapping source queries/models, not independent replications. Subgroups selected by observed recurring mean harm are posthoc; no p values, causal or promotion claim. Dispersion compression and rank correlation do not identify encoder versus readout or prove missing information. All groups retained; insufficient halves explicitly null. No outer labels used in offset fitting.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

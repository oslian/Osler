"""Expected within-half permutation loss at fixed prediction marginals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_source_group_residuals import crossfit_offset, patient_half
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def independent_pair_mae(truth, prediction):
    """Exact mean of all pairwise absolute errors, without an n-by-n matrix."""
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or not len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite nonempty vectors required')
    # Center before cumulative sums to limit cancellation at large offsets.
    center = np.median(y)
    y = np.sort(y-center)
    p = p-center
    prefix = np.concatenate(([0.], np.cumsum(y)))
    count = np.searchsorted(y, p, side='left')
    distance = p*count-prefix[count]+prefix[-1]-prefix[count]-p*(len(y)-count)
    return float(distance.mean()/len(y))


def assignment_metrics(truth, prediction, halves):
    y, p, h = np.asarray(truth, float), np.asarray(prediction, float), np.asarray(halves)
    corrected = crossfit_offset(y, p, h)
    if corrected is None:
        return None
    actual = float(abs(corrected-y).mean())
    independent = sum(independent_pair_mae(y[h == half], corrected[h == half]) * int((h == half).sum())
                      for half in (0, 1))/len(y)
    return dict(actual_mae=actual, expected_permuted_mae=float(independent),
                assignment_benefit=float(independent-actual))


def main():
    output = Path('artifacts/native_reconstruction_source_assignment_20261001.json')
    if output.exists():
        raise ValueError('Existing diagnosis')
    parent_path = Path('artifacts/native_reconstruction_source_group_residuals_20261001.json')
    parent = json.loads(parent_path.read_text())
    checked = {}
    verify_bound_files(parent['bindings'], checked)
    root = Path('data/research_recovery/additional_development_domains_20260921')
    paths = [Path(__file__), parent_path, Path('scripts/diagnose_source_group_residuals.py'),
             Path('scripts/score_monitored_continuation_outer.py')]
    folds = {}
    for fold in range(5):
        base_path = root / f'coverage_selection_20260926/fold{fold}.parquet'
        fresh_path = root / f'continuation_selection_20260930/fold{fold}.parquet'
        base, fresh = pd.read_parquet(base_path), pd.read_parquet(fresh_path)
        pd.testing.assert_frame_equal(base[['patient', 'target', 'fold']], fresh[['patient', 'target', 'fold']])
        data = base.copy()
        for arm in ('real', 'process'):
            data[arm] = fresh[arm+'10'].to_numpy()
        results = {}
        for (target, component), group in data.groupby(['target', 'component'], sort=True):
            h = np.asarray([patient_half(p) for p in group.patient])
            y = group.truth.to_numpy()/group.fit_scale.to_numpy()
            stats = {arm: assignment_metrics(y, group[arm].to_numpy()/group.fit_scale.to_numpy(), h)
                     for arm in ('real', 'process', 'population')}
            reference = parent['folds'][str(fold)][target][str(component)]
            for arm in stats:
                expected = reference['arms'][arm]['crossfit_local_offset_mae']
                if (stats[arm] is None) != (expected is None):
                    raise ValueError('Changed calibration support')
                if stats[arm] is not None and not np.isclose(stats[arm]['actual_mae'], expected, atol=1e-12, rtol=1e-12):
                    raise ValueError('Changed corrected predictions')
            if stats['population'] is not None and not np.isclose(stats['population']['assignment_benefit'], 0., atol=1e-10):
                raise ValueError('Population reference is not constant within a half')
            results.setdefault(target, {})[str(component)] = dict(patients=len(group), arms=stats,
                recurring_mean_harm=reference['recurring_mean_harm'])
        folds[str(fold)] = results
        if len(results) != 47:
            raise ValueError('Incomplete target family')
        paths.extend((base_path, fresh_path))
        print('Assignment contrast fold', fold, flush=True)
    summary = {}
    for control in ('process', 'population'):
        rows = [r for fold in folds.values() for groups in fold.values() for r in groups.values()
                if r['patients'] >= 30 and r['recurring_mean_harm'][control] and r['arms']['real'] is not None]
        residual = [r for r in rows if r['arms']['real']['actual_mae'] > r['arms'][control]['actual_mae']]
        summary[control] = dict(
            supported_recurring_harm_groups=len(rows),
            matched_local_harm_groups=len(residual),
            matched_local_harm_but_assignment_benefit_groups=sum(r['arms']['real']['assignment_benefit'] > 1e-12 for r in residual),
            median_real_assignment_benefit=float(np.median([r['arms']['real']['assignment_benefit'] for r in rows])),
            median_control_assignment_benefit=float(np.median([r['arms'][control]['assignment_benefit'] for r in rows])))
    output.write_text(json.dumps(dict(summary=summary, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, promotion_allowed=False, runtime_modified=False,
        limitation='Expected output permutation loss within the same source component and held-out calibration half, holding prediction marginals/calibration fixed. Not input placebo, retraining, significance, latent information ceiling or an external generalization gate. Overlapping folds and posthoc recurring-harm subsets. Positive assignment benefit can coexist with worse MAE than a constant baseline. No deployment calibration, future forecast or causal claim.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Test global glucose output scale without changing the shared encoder."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/glucose_native_calibration_20260911.json')
    reference = Path('artifacts/rate_glucose_jepa_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(reference.read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('changed original input')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    labels = np.load(root / 'multitarget_labels.npz')
    k = list(labels['variables']).index('glucose')
    frame = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    frame = frame.rename(columns={'uniquepid': 'subject_id'})
    y, current = labels['future'][:, k], labels['current'][:, k]
    frame['truth'] = y
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        points = dict(np.load(root / f'glucose_native_predictions_{seed}.npz'))
        partitions = np.load(root / f'partition_{seed}.npz')
        roles = {role: partitions[role][valid] & labels['paired'][:, k]
                 for role in ('tune', 'evaluation', 'hospital')}
        tune = roles['tune']
        subjects = frame.loc[tune, 'subject_id']
        weight = 1. / subjects.map(subjects.value_counts()).to_numpy()
        coefficients = {}
        for arm in ('own', 'real'):
            delta = points[arm] - current
            residual = y[tune] - points[arm][tune]
            order = np.argsort(residual)
            offset = float(residual[order][np.searchsorted(np.cumsum(weight[order]), weight.sum() / 2)])
            model = QuantileRegressor(alpha=0., quantile=.5, solver='highs')
            model.fit(delta[tune, None], y[tune] - current[tune], sample_weight=weight)
            intercept, gain = float(model.intercept_), float(model.coef_[0])
            coefficients[arm] = dict(offset=offset, intercept=intercept, gain=gain)
            points[f'{arm}_offset'] = points[arm] + offset
            points[f'{arm}_affine'] = current + intercept + gain * delta
        item = dict(seed=seed, coefficients=coefficients, roles={})
        for role in ('evaluation', 'hospital'):
            mask = roles[role]
            work = frame.loc[mask]
            mae = {arm: patient_equal_mae(work, y[mask], point[mask]) for arm, point in points.items()}
            for arm in ('own', 'real', 'support', 'persistence'):
                np.testing.assert_allclose(mae[arm], entry['roles'][role]['mae'][arm], atol=1e-7, rtol=0)
            comparisons = {}
            for mode in ('offset', 'affine'):
                for a, b in ((f'own_{mode}', 'own'), (f'real_{mode}', 'real'),
                             (f'real_{mode}', f'own_{mode}')):
                    comparisons[f'{a}_vs_{b}'] = paired(work, points[a][mask], points[b][mask],
                                                       f'glucose-scale:{seed}:{role}:{a}:{b}', 2000)
            diagnostics = {}
            for arm in ('own', 'real'):
                error = points[arm][mask] - y[mask]
                bias = pd.Series(error, index=work.subject_id).groupby(level=0).mean().mean()
                corr = np.corrcoef(points[arm][mask] - current[mask], y[mask] - current[mask])[0, 1]
                diagnostics[arm] = dict(patient_mean_bias=float(bias), row_delta_correlation=float(corr))
            item['roles'][role] = dict(mae=mae, comparisons=comparisons, diagnostics=diagnostics)
        results.append(item)
    paths = (reference, Path(__file__), Path('GLUCOSE_NATIVE_CALIBRATION_PROTOCOL_20260911.md'))
    output.write_text(json.dumps(dict(complete=True, status='diagnostic_only', runtime_modified=False,
        binding={str(p): sha256(p) for p in paths}, seeds=results), indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        group = [r['roles'][role] for r in results]
        print(role, {a: round(float(np.median([g['mae'][a] for g in group])), 5) for a in group[0]['mae']})
        for key in group[0]['comparisons']:
            ci = [g['comparisons'][key]['ci95'] for g in group]
            print(key, 'gain/harm', sum(c[1] < 0 for c in ci), sum(c[0] > 0 for c in ci))
        print('diagnostics', {a: {m: round(float(np.median([g['diagnostics'][a][m] for g in group])), 4)
              for m in group[0]['diagnostics'][a]} for a in ('own', 'real')})


if __name__ == '__main__':
    main()

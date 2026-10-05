"""Fixed patient-equal ablation of the frozen-point residual corrector."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from whole_body_jepa.multisource_intervals import sha256


def predict(source, query):
    if (source.groupby('component').size() >= 30).sum() < 3:
        return None, 'insufficient_support'
    if source.fit_scale.nunique() != 1 or not (source.fit_scale > 0).all():
        raise ValueError('Invalid scale')
    scale = float(source.fit_scale.iloc[0])
    weights = np.full(len(source), 1 / len(source))
    scaler = StandardScaler().fit(source[['real']], sample_weight=weights)
    x = scaler.transform(source[['real']])
    y = (source.truth - source.real).to_numpy() / scale
    model = HuberRegressor(epsilon=1.35, alpha=.1, max_iter=300, tol=1e-5)
    for budget in (300, 2700):
        model.set_params(max_iter=budget, warm_start=budget != 300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y, sample_weight=weights)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if converged:
            break
    result = query.real.to_numpy() + scale * model.predict(scaler.transform(query[['real']]))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction')
    return result, 'fitted' if converged else 'nonconverged'


def main():
    prior = Path('artifacts/native_reconstruction_point_only_20260925.json')
    reference = json.loads(prior.read_text())
    root = Path('data/research_recovery/additional_development_domains_20260921/calibration_20260924')
    paths = [root / f'fold{f}.parquet' for f in range(5)]
    frames = [pd.read_parquet(p) for p in paths]
    results = []
    with threadpool_limits(limits=4):
        for target, folds in reference['targets'].items():
            for fold, old in folds.items():
                source = frames[int(fold)].loc[frames[int(fold)].target == target]
                if source.patient.duplicated().any():
                    raise ValueError('Duplicate patient')
                rows = []
                for component in old['components']:
                    mask = source.component.astype(str) == component['component']
                    query = source.loc[mask]
                    if len(query) != component['patients']:
                        raise ValueError('Changed query support')
                    p, status = predict(source.loc[~mask], query)
                    rows.append(dict(component=component['component'], patients=len(query), status=status,
                                     mae=float(np.mean(abs(p-query.truth.to_numpy()))) if p is not None else None,
                                     balanced_mae=component['mae'],
                                     unchanged_mae=component['reference_mae']['unchanged']))
                matched = bool(rows) and old['complete'] and all(r['status'] == 'fitted' for r in rows)
                record = dict(target=target, fold=fold, matched=matched, components=rows)
                if matched:
                    record['means'] = {level: {a: float(np.average([r[a] for r in rows], weights=w))
                                             for a in ('mae', 'balanced_mae', 'unchanged_mae')}
                                       for level, w in [('component', None), ('patient', [r['patients'] for r in rows])]}
                results.append(record)
            print('Completed', target, flush=True)
    matched = [r for r in results if r['matched']]
    summary = {level: {ref: sum(r['means'][level]['mae'] < r['means'][level][ref] for r in matched)
                       for ref in ('balanced_mae', 'unchanged_mae')}
               for level in ('component', 'patient')}
    summary['matched'] = len(matched)
    output = Path('artifacts/native_reconstruction_patient_equal_20260925.json')
    if output.exists():
        raise ValueError('Existing result')
    output.write_text(json.dumps(dict(summary=summary, records=results,
        bindings={str(p): sha256(p) for p in [prior, Path(__file__), *paths]},
        runtime_modified=False, promotion_allowed=False, outer_evaluation_performed=False), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

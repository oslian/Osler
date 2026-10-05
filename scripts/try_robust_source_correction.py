"""Fixed robust source-only LOCO follow-up; does not open outer predictions."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.try_conditional_reconstruction_repair import features, weights
from whole_body_jepa.multisource_intervals import sha256


def robust_predict(source, query, variables):
    if (source.groupby('component').size() >= 30).sum() < 3:
        return {}, dict(status='insufficient_support')
    process, numeric, selected = features(source, query, variables)
    w = weights(source)
    if source.fit_scale.nunique() != 1 or not (source.fit_scale > 0).all():
        raise ValueError('Invalid scale')
    scale = float(source.fit_scale.iloc[0])
    y = (source.truth-source.real).to_numpy()/scale
    pred = {}; fits = {}
    for name, x in (('process_context', process), ('numeric_context', numeric)):
        scaler = StandardScaler().fit(x['source'], sample_weight=w)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            model = HuberRegressor(epsilon=1.35, alpha=.1, max_iter=300, tol=1e-5)
            model.fit(scaler.transform(x['source']), y, sample_weight=w)
        bad = any(issubclass(z.category, ConvergenceWarning) for z in captured)
        prediction = query.real.to_numpy()+scale*model.predict(scaler.transform(x['query']))
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite robust prediction')
        fits[name] = dict(converged=not bad, iterations=int(model.n_iter_), residual_scale=float(model.scale_))
        pred[name] = prediction
    return pred, dict(status='fitted' if all(x['converged'] for x in fits.values()) else 'nonconverged',
                      input_variables=selected, arms=fits)


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_conditional_repair_20260925.json')
    sp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    protocol = Path('RECONSTRUCTION_ROBUST_SOURCE_PROTOCOL_20260925.md')
    out = Path('artifacts/native_reconstruction_robust_source_20260925.json')
    if out.exists():
        raise ValueError('Existing experiment')
    old = json.loads(rp.read_text()); s = json.loads(sp.read_text())
    snapshot = root/'variable_snapshot_20260925.parquet'
    if sha256(snapshot) != s['private_sha256']:
        raise ValueError('Changed snapshot')
    frames = {}; paths = [rp, sp, snapshot, protocol, Path(__file__), Path('scripts/try_conditional_reconstruction_repair.py')]
    for f in range(5):
        p = root/f'calibration_20260924/fold{f}.parquet'
        meta = json.loads(p.with_suffix('.json').read_text())
        if sha256(p) != meta['prediction_sha256']:
            raise ValueError('Changed calibration')
        paths.extend((p, p.with_suffix('.json')))
        frames[f] = pd.read_parquet(p)
    results = {}
    with threadpool_limits(limits=4):
        for i, target in enumerate(old['targets']):
            snap = pd.read_parquet(snapshot, filters=[('target', '=', target)])
            results[target] = {}
            for f in range(5):
                source = frames[f].loc[frames[f].target == target].merge(snap, on=['patient', 'target'], how='left', validate='one_to_one', indicator=True)
                if not (source._merge == 'both').all():
                    raise ValueError('Missing snapshot')
                prior = {r['component']: r for r in old['fits'][target][str(f)]['source_cv']['components']}
                rows = []
                for component, query in source.groupby('component', sort=True):
                    pred, info = robust_predict(source.loc[source.component != component], query, s['variables'])
                    ref = prior[str(component)]
                    if ref['patients'] != len(query):
                        raise ValueError('Changed CV identity')
                    rows.append(dict(component=str(component), patients=len(query), fit=info,
                        reference_mae=ref['mae'], robust_mae={a: float(np.mean(abs(p-query.truth.to_numpy()))) for a,p in pred.items()}))
                complete = bool(rows) and all(r['fit']['status'] == 'fitted' for r in rows)
                result = dict(complete=complete, components=rows)
                if complete:
                    result['robust_mae'] = {a: float(np.mean([r['robust_mae'][a] for r in rows])) for a in ('process_context','numeric_context')}
                    result['reference_mae'] = {a: float(np.mean([r['reference_mae'][a] for r in rows])) for a in ('unchanged','offset','process_context','numeric_context')}
                results[target][str(f)] = result
            print('Robust source target', i+1, 'of', len(old['targets']), flush=True)
    complete = [r for z in results.values() for r in z.values() if r['complete']]
    summary = dict(target_folds=235, complete_target_folds=len(complete),
        numeric_better_than_unchanged=sum(r['robust_mae']['numeric_context'] < r['reference_mae']['unchanged'] for r in complete),
        numeric_better_than_ridge=sum(r['robust_mae']['numeric_context'] < r['reference_mae']['numeric_context'] for r in complete),
        numeric_better_than_all_references=sum(r['robust_mae']['numeric_context'] < min(*r['reference_mae'].values(), r['robust_mae']['process_context']) for r in complete),
        nonconverged_component_fits=sum(c['fit']['status']=='nonconverged' for z in results.values() for r in z.values() for c in r['components']))
    out.write_text(json.dumps(dict(summary=summary, targets=results, bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False, promotion_allowed=False, outer_evaluation_performed=False,
        limitation='Fixed robust source LOCO. Same frozen current reconstruction; descriptive means not independent promotion statistics. Robust loss has fitted scale and differs from ridge regularization semantics.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

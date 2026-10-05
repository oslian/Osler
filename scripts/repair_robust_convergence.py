"""Same objective continuation for five recorded nonconverged source fits."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.diagnose_seen_domain_correction import query_people, matched_fits
from scripts.try_conditional_reconstruction_repair import features, weights
from whole_body_jepa.multisource_intervals import sha256


def solve_more(x, y, w, query):
    model = HuberRegressor(epsilon=1.35, alpha=.1, max_iter=300, tol=1e-5)
    records = []
    for stage in ('original', 'continued'):
        if stage == 'continued':
            model.set_params(warm_start=True, max_iter=2700)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x, y, sample_weight=w)
        predicted = model.predict(query)
        if not np.isfinite(predicted).all():
            raise ValueError('Nonfinite optimization result')
        records.append(dict(stage=stage, converged=not any(issubclass(c.category, ConvergenceWarning) for c in caught),
                            iterations=int(model.n_iter_), scale=float(model.scale_),
                            coefficients_l2=float(np.linalg.norm(model.coef_)), prediction=predicted))
    return records


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_seen_domain_20260925.json')
    sp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    out = Path('artifacts/native_reconstruction_robust_convergence_20260925.json')
    if out.exists():
        raise ValueError('Existing numerical diagnosis')
    r = json.loads(rp.read_text()); s = json.loads(sp.read_text())
    for p,h in r['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed dependency')
    selected = [(t,int(f),c) for t,z in r['targets'].items() for f,x in z.items()
                for c in x['components'] if c.get('seen_fit',{}).get('status')=='nonconverged']
    if len(selected) != 5:
        raise ValueError('Changed failure scope')
    results = []
    with threadpool_limits(limits=4):
        for target,fold,prior in selected:
            p = root/f'calibration_20260924/fold{fold}.parquet'
            meta = json.loads(p.with_suffix('.json').read_text())
            if sha256(p) != meta['prediction_sha256']:
                raise ValueError('Changed calibration')
            source = pd.read_parquet(p);queries=query_people(source)
            snap = pd.read_parquet(root/'variable_snapshot_20260925.parquet', filters=[('target','=',target)])
            source = source.loc[source.target==target].merge(snap,on=['patient','target'],validate='one_to_one')
            components = {str(c):c for c in source.component.unique()}
            seen,_,query = matched_fits(source,queries,components[prior['component']])
            _,numeric,variables = features(seen,query,s['variables'])
            if variables != prior['seen_fit']['input_variables'] or len(seen) != prior['matched_fit_patients']:
                raise ValueError('Changed exact fit inputs')
            w=weights(seen);scale=float(seen.fit_scale.iloc[0])
            scaler=StandardScaler().fit(numeric['source'],sample_weight=w)
            records=solve_more(scaler.transform(numeric['source']), (seen.truth-seen.real).to_numpy()/scale,
                               w, scaler.transform(numeric['query']))
            original=prior['seen_fit']['arms']['numeric_context']
            if records[0]['converged'] or records[0]['iterations']!=original['iterations'] or not np.isclose(records[0]['scale'],original['residual_scale'],atol=1e-12,rtol=1e-9):
                raise ValueError('Original nonconvergence did not reproduce')
            before=query.real.to_numpy()+scale*records[0]['prediction']
            after=query.real.to_numpy()+scale*records[1]['prediction']
            for record in records:
                record.pop('prediction')
            results.append(dict(target=target,fold=fold,component=prior['component'],fit_patients=len(seen),
                numeric_features=numeric['source'].shape[1],query_patients=len(query),stages=records,
                query_mae=dict(unchanged=float(np.mean(abs(query.real-query.truth))),
                               original=float(np.mean(abs(before-query.truth))), continued=float(np.mean(abs(after-query.truth)))),
                mean_prediction_shift=float(np.mean(abs(after-before))),
                fit_truth_mode_fraction=float(seen.truth.value_counts(normalize=True).iloc[0])))
            print('Continued numerical fit',len(results),'of',len(selected),flush=True)
    summary=dict(attempted=len(results),converged=sum(x['stages'][1]['converged'] for x in results),
        original_failures_reproduced=True,method='Same objective, original300 then warm-start2700 maximum iterations; no tolerance change')
    out.write_text(json.dumps(dict(summary=summary,results=results,bindings={str(p):sha256(p) for p in (rp,sp,Path(__file__))},
        runtime_modified=False,promotion_allowed=False,outer_evaluation_performed=False,
        limitation='Numerical repair only for prespecified failed source fits. Query errors descriptive, not selection or promotion. Data-support failures unchanged.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

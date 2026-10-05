"""Source-only numeric advantage versus fixed observation-support summaries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from whole_body_jepa.multisource_intervals import sha256


def association(frame, feature):
    part = frame[[feature, 'advantage', 'component']].dropna()
    if len(part) < 30 or part[feature].nunique() < 2 or part.advantage.nunique() < 2:
        return dict(patients=len(part), within_component_rho=None)
    centered = part[[feature, 'advantage']] - part.groupby('component')[[feature, 'advantage']].transform('mean')
    rho = float(spearmanr(centered[feature], centered.advantage).statistic)
    return dict(patients=len(part), within_component_rho=rho if np.isfinite(rho) else None)


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    report = json.loads(rp.read_text())
    snapshot = root/'variable_snapshot_20260925.parquet'
    if sha256(snapshot) != report['private_sha256']:
        raise ValueError('Changed snapshots')
    variables = report['variables']
    masks = [f'observed_{v}' for v in variables]
    ages = [f'age_{v}' for v in variables]
    predictions, bindings = {}, {str(rp):sha256(rp),str(snapshot):sha256(snapshot)}
    for fold in range(5):
        p = root/f'coverage_selection_20260926/fold{fold}.parquet'
        meta = json.loads(p.with_suffix('.json').read_text())
        if sha256(p) != meta['prediction_sha256']:
            raise ValueError('Changed predictions')
        predictions[fold] = pd.read_parquet(p)
        bindings[str(p)] = sha256(p)
    results = {}
    for target in sorted(predictions[0].target.unique()):
        snap = pd.read_parquet(snapshot, columns=['patient','target',*masks,*ages],filters=[('target','=',target)])
        observed = snap[masks].to_numpy(bool)
        age = snap[ages].to_numpy(float)
        if not np.isfinite(age[observed]).all() or (age[observed]<0).any():
            raise ValueError('Invalid observed ages')
        features = snap[['patient','target']].copy()
        count = observed.sum(1)
        features['observed_count'] = count
        features['fresh_1h_count'] = (observed & (age<=1+1e-6)).sum(1)
        features['fresh_6h_count'] = (observed & (age<=6+1e-6)).sum(1)
        features['mean_observed_age'] = np.divide(np.where(observed,age,0).sum(1),count,out=np.full(len(count),np.nan),where=count>0)
        results[target] = {}
        for fold,pred in predictions.items():
            part = pred.loc[pred.target.eq(target)].merge(features,on=['patient','target'],validate='one_to_one',how='left',indicator=True)
            if not part._merge.eq('both').all():
                raise ValueError('Missing support snapshot')
            part['advantage'] = (abs(part.process-part.truth)-abs(part.real-part.truth))/part.fit_scale
            results[target][str(fold)] = {f:association(part,f) for f in ['observed_count','fresh_1h_count','fresh_6h_count','mean_observed_age']}
        print('Support',target,flush=True)
    summary = {}
    for feature in ['observed_count','fresh_1h_count','fresh_6h_count','mean_observed_age']:
        vals = [r[feature]['within_component_rho'] for t in results.values() for r in t.values() if r[feature]['within_component_rho'] is not None]
        summary[feature] = dict(comparisons=len(vals),median_rho=float(np.median(vals)),
                               positive=sum(x>0 for x in vals),absolute_at_least_point_one=sum(abs(x)>=.1 for x in vals))
    dest = Path('artifacts/native_reconstruction_coverage_support_20260929.json')
    dest.write_text(json.dumps(dict(summary=summary,targets=results,bindings=bindings,
        promotion_allowed=False,runtime_modified=False,
        limitation='Source-selection only; descriptive centered rank associations, no significance or causal interpretation. Overlapping patients/fits. No support-based tuning or exclusion.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

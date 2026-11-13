"""One-feature Huber residual calibration versus the existing full correction."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.try_conditional_reconstruction_repair import weights
from whole_body_jepa.multisource_intervals import sha256


def point_only(source, query):
    if (source.groupby('component').size()>=30).sum()<3:
        return None,dict(status='insufficient_support')
    w=weights(source)
    if source.fit_scale.nunique()!=1 or not (source.fit_scale>0).all():raise ValueError('Invalid scale')
    scale=float(source.fit_scale.iloc[0]);y=(source.truth-source.real).to_numpy()/scale
    scaler=StandardScaler().fit(source[['real']],sample_weight=w)
    x=scaler.transform(source[['real']]);q=scaler.transform(query[['real']])
    model=HuberRegressor(epsilon=1.35,alpha=.1,max_iter=300,tol=1e-5)
    continued=False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(x,y,sample_weight=w)
    if any(issubclass(c.category,ConvergenceWarning) for c in caught):
        continued=True;model.set_params(warm_start=True,max_iter=2700)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(x,y,sample_weight=w)
    converged=not any(issubclass(c.category,ConvergenceWarning) for c in caught)
    pred=query.real.to_numpy()+scale*model.predict(q)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite point correction')
    return pred,dict(status='fitted' if converged else 'nonconverged',continued=continued,
        iterations=int(model.n_iter_),normalized_slope=float(model.coef_[0]),residual_scale=float(model.scale_))


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    rp=Path('artifacts/native_reconstruction_robust_source_20260925.json')
    out=Path('artifacts/native_reconstruction_point_only_20260925.json')
    if out.exists():raise ValueError('Existing point-only diagnosis')
    old=json.loads(rp.read_text())
    for p,h in old['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed comparison dependency')
    frames={f:pd.read_parquet(root/f'calibration_20260924/fold{f}.parquet') for f in range(5)}
    results={}
    with threadpool_limits(limits=4):
        for i,t in enumerate(old['targets']):
            results[t]={}
            for f in range(5):
                source=frames[f].loc[frames[f].target==t]
                reference={r['component']:r for r in old['targets'][t][str(f)]['components']}
                rows=[]
                for comp,query in source.groupby('component',sort=True):
                    pred,info=point_only(source.loc[source.component!=comp],query)
                    prior=reference[str(comp)]
                    if len(query)!=prior['patients']:raise ValueError('Changed comparison rows')
                    rows.append(dict(component=str(comp),patients=len(query),fit=info,
                        mae=float(np.mean(abs(pred-query.truth.to_numpy()))) if pred is not None else None,
                        reference_mae=prior['reference_mae'],full_robust_mae=prior['robust_mae']))
                complete=bool(rows) and all(r['fit']['status']=='fitted' for r in rows)
                matched=complete and old['targets'][t][str(f)]['complete']
                record=dict(complete=complete,matched=matched,components=rows)
                if complete:record['point_mae']=float(np.mean([r['mae'] for r in rows]))
                if matched:
                    record['comparators']={**old['targets'][t][str(f)]['reference_mae'],
                        'full_robust':old['targets'][t][str(f)]['robust_mae']['numeric_context'],
                        'process_robust':old['targets'][t][str(f)]['robust_mae']['process_context']}
                results[t][str(f)]=record
            print('Point-only source target',i+1,'of',len(old['targets']),flush=True)
    matched=[r for z in results.values() for r in z.values() if r['matched']]
    summary=dict(target_folds=235,complete=sum(r['complete'] for z in results.values() for r in z.values()),
        matched_complete=len(matched),better_than_full=sum(r['point_mae']<r['comparators']['full_robust'] for r in matched),
        better_than_unchanged=sum(r['point_mae']<r['comparators']['unchanged'] for r in matched),
        better_than_offset=sum(r['point_mae']<r['comparators']['offset'] for r in matched),
        better_than_every_reference=sum(r['point_mae']<min(r['comparators'].values()) for r in matched))
    out.write_text(json.dumps(dict(summary=summary,targets=results,bindings={str(p):sha256(p) for p in (rp,Path(__file__))},
        runtime_modified=False,promotion_allowed=False,outer_evaluation_performed=False,
        limitation='Fixed one-feature Huber source LOCO diagnostic. Same objective/weights/splits as full robust arm; descriptive means, not promotion. Current reconstruction only.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

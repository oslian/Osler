"""Common-support residual comparison, not model selection or causal matching."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


FEATURES=('fresh_1h_variables','median_latest_age_hours','history_span_hours')


def strata(source, frame):
    cuts={'real':np.unique(np.quantile(source.real,[1/3,2/3])).tolist()}
    cuts.update({f:[float(source[f].median())] for f in FEATURES})
    ids=np.zeros(len(frame),dtype=int);factor=1
    for name,bounds in cuts.items():
        ids+=factor*np.searchsorted(bounds,frame[name].to_numpy(),side='right')
        factor*=len(bounds)+1
    return ids,cuts


def table(frame, ids):
    sizes=frame.groupby('component').size()
    w=(1/frame.component.map(sizes)).to_numpy();w=w/w.sum()
    residual=(frame.truth-frame.real).to_numpy()
    rows={}
    for b in np.unique(ids):
        m=ids==b;p=frame.loc[m]
        rows[int(b)]=dict(patients=len(p),supported_components=int((p.groupby('component').size()>=30).sum()),
            mass=float(w[m].sum()),mean_residual=float(np.average(residual[m],weights=w[m])))
    return rows


def diagnose(source, outer):
    if source.empty or outer.empty:return dict(status='empty_role')
    a_ids,cuts=strata(source,source);b_ids,_=strata(source,outer)
    a=table(source,a_ids);b=table(outer,b_ids)
    common=sorted(set(a)&set(b))
    accepted=[k for k in common if a[k]['supported_components']>=3 and b[k]['supported_components']>=3]
    result=dict(status='supported' if accepted else 'no_supported_overlap',cuts=cuts,
        source_strata=a,outer_strata=b,common_strata=len(common),supported_strata=len(accepted),
        source_retained_patient_fraction=float(np.isin(a_ids,accepted).mean()),
        outer_retained_patient_fraction=float(np.isin(b_ids,accepted).mean()),
        source_retained_component_mass=sum(a[k]['mass'] for k in accepted),
        outer_retained_component_mass=sum(b[k]['mass'] for k in accepted))
    if not accepted:return result
    weights=np.array([min(a[k]['mass'],b[k]['mass']) for k in accepted]);weights/=weights.sum()
    shifts=np.array([b[k]['mean_residual']-a[k]['mean_residual'] for k in accepted])
    result.update(matched_signed_shift=float(weights@shifts),matched_absolute_bin_shift=float(weights@abs(shifts)))
    # Reference uses EXACTLY the same retained people, with original role component weights.
    ma=sum(a[k]['mass'] for k in accepted);mb=sum(b[k]['mass'] for k in accepted)
    result['same_retained_unmatched_shift']=sum(b[k]['mass']*b[k]['mean_residual'] for k in accepted)/mb-sum(a[k]['mass']*a[k]['mean_residual'] for k in accepted)/ma
    return result


def main():
    rp=Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    sp=Path('artifacts/native_reconstruction_additional_observation_support_20260925.json')
    r=json.loads(rp.read_text());s=json.loads(sp.read_text())
    for p,h in {**r['bindings'],**s['bindings']}.items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    root=Path('data/research_recovery/additional_development_domains_20260921')
    support_path=root/'observation_support_20260925.parquet'
    if sha256(support_path)!=s['private_sha256']:raise ValueError('Changed support')
    support=pd.read_parquet(support_path);results={}
    for fold in range(5):
        frames=[]
        for role in ('calibration','outer'):
            raw=pd.read_parquet(root/f'{role}_20260924/fold{fold}.parquet')
            frame=raw.merge(support,on=['patient','target'],how='left',validate='one_to_one',indicator=True)
            if not (frame._merge=='both').all():raise ValueError('Missing history support')
            frames.append(frame)
        results[str(fold)]={t:diagnose(frames[0].loc[frames[0].target==t],frames[1].loc[frames[1].target==t]) for t in r['targets']}
    evaluated=[x for f in results.values() for x in f.values() if x['status']=='supported']
    broad=[x for x in evaluated if min(x['source_retained_patient_fraction'],x['outer_retained_patient_fraction'])>=.8]
    summary=dict(target_folds=235,any_supported_overlap=len(evaluated),both_roles_at_least_80pct_retained=len(broad),
        broad_signed_shift_reduced=sum(abs(x['matched_signed_shift'])<abs(x['same_retained_unmatched_shift']) for x in broad))
    out=Path('artifacts/native_reconstruction_support_matched_shift_20260925.json')
    if out.exists():raise ValueError('Existing diagnosis')
    out.write_text(json.dumps(dict(summary=summary,folds=results,bindings={str(p):sha256(p) for p in (rp,sp,support_path,Path(__file__))},
        runtime_modified=False,promotion_allowed=False,
        limitation='Posthoc common-support diagnostic. Source-only predictor/support bin edges, but overlap uses both covariate distributions. Not deployable, not causal, not exact physiology matching. Means not MAE. Retained support mandatory to interpret.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

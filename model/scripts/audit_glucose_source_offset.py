"""Source-known offset correction with fixed encoder and shuffled-source control."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/glucose_source_offset_20260911.json')
    reference=Path('artifacts/rate_glucose_jepa_20260911.json')
    if out.exists(): raise ValueError('fresh output required')
    ref=json.loads(reference.read_text())
    for name,digest in ref['binding'].items():
        if sha256(Path(name))!=digest: raise ValueError('input changed')
    meta=pd.read_parquet(root/'glucose_source_metadata.parquet')
    raw=pd.read_parquet(root/'body_labs.parquet')
    raw=raw.loc[raw.variable.eq('glucose')].merge(meta[['source_id','labname']],on='source_id',validate='one_to_one')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    l=np.load(root/'multitarget_labels.npz');k=list(l['variables']).index('glucose');eligible=l['paired'][:,k]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=l['future'][:,k]
    source=np.full(len(q),np.nan);groups={s:g for s,g in raw.groupby('stay_id')}
    for i in np.flatnonzero(eligible):
        row=q.iloc[i];g=groups[row.stay_id];measured=row.t-pd.Timedelta(hours=float(l['measurement_age'][i,k]))
        match=g.loc[(g.available_at<=row.t)&((g.measured_at-measured).dt.total_seconds().abs()<.01)&np.isclose(g.value,l['current'][i,k],atol=1e-5,rtol=0)]
        names=match.labname.unique()
        if len(names)==1: source[i]=float(names[0]=='bedside glucose')
    if not np.isfinite(source[eligible]).all(): raise ValueError('unresolved current source')
    results=[]
    for entry in ref['seeds']:
        seed=entry['seed'];points=dict(np.load(root/f'glucose_native_predictions_{seed}.npz'))
        partitions=np.load(root/f'partition_{seed}.npz')
        roles={r:partitions[r][valid]&eligible for r in ('tune','evaluation','hospital')}
        shuffled=source.copy();rng=np.random.default_rng(seed)
        for mask in roles.values(): shuffled[mask]=rng.permutation(source[mask])
        tune=roles['tune'];ids=q.loc[tune,'subject_id'];weight=1/ids.map(ids.value_counts()).to_numpy()
        coeff={}
        for arm in ('own','real'):
            residual=q.truth.to_numpy()[tune]-points[arm][tune]
            for name,feature in [('source',source),('shuffled',shuffled),('constant',np.zeros(len(q)))]:
                model=QuantileRegressor(alpha=0,quantile=.5,solver='highs').fit(feature[tune,None],residual,sample_weight=weight)
                key=f'{arm}_{name}';points[key]=points[arm]+model.intercept_+model.coef_[0]*feature
                coeff[key]=dict(intercept=float(model.intercept_),source_effect=float(model.coef_[0]))
        item=dict(seed=seed,coefficients=coeff,roles={})
        for role in ('evaluation','hospital'):
            mask=roles[role];f=q.loc[mask]
            mae={a:patient_equal_mae(f,f.truth,p[mask]) for a,p in points.items()}
            for arm in ('own','real','support','persistence'):
                np.testing.assert_allclose(mae[arm],entry['roles'][role]['mae'][arm],rtol=0,atol=1e-7)
            contrasts=[('real_source','real'),('real_source','real_constant'),('real_source','real_shuffled'),('real_source','own_source')]
            item['roles'][role]=dict(mae=mae,comparisons={a+'_vs_'+b:paired(f,points[a][mask],points[b][mask],f'source-offset:{seed}:{role}:{a}:{b}',2000) for a,b in contrasts})
        results.append(item)
    paths=(reference,root/'glucose_source_metadata.parquet',Path(__file__))
    out.write_text(json.dumps(dict(complete=True,status='candidate_only',runtime_modified=False,future_source_input=False,
        binding={str(p):sha256(p) for p in paths},seeds=results),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        z=[s['roles'][role] for s in results]
        print(role,{a:round(float(np.median([g['mae'][a] for g in z])),5) for a in ('own_source','real','real_source','real_constant','real_shuffled')})
        for c in z[0]['comparisons']:
            print(c,'gain/harm',sum(g['comparisons'][c]['ci95'][1]<0 for g in z),sum(g['comparisons'][c]['ci95'][0]>0 for g in z))


if __name__=='__main__': main()

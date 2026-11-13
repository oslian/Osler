"""Fit-only patient-equal scaling from exact accepted current anchor labels."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import fit_scaler
from whole_body_jepa.multisource_intervals import sha256


def anchor_labels(events, queries):
    """Exact anchor label for canonical single-version sources; reject revisions.

    Full-history conflict replay remains authoritative. This faster extraction
    supports this raw ledger only after checking each source has one version.
    """
    events=events.loc[events.source_kind.isin(('lab_available','bedside_available'))].copy()
    identity=['stay_id','source_kind','source_id']
    if events.duplicated(identity).any():
        raise ValueError('Multiple source versions require full replay')
    if events[['event_time','measured_at','value']].isna().any().any():
        raise ValueError('Invalid source values/times')
    if not np.isfinite(events.value).all() or (events.measured_at>events.event_time).any():
        raise ValueError('Invalid source values/times')
    q=queries[['stay_id','variable','t']].reset_index(drop=True).copy()
    q['query_id']=np.arange(len(q))
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='2GB'")
    db.register('events',events);db.register('queries',q)
    result=db.sql("""WITH candidates AS (
        SELECT q.query_id,q.t,e.stay_id,e.variable,e.measured_at,e.value,e.quality_valid
        FROM queries q JOIN events e ON q.stay_id=e.stay_id
            AND q.variable=e.variable AND q.t=e.event_time
        LEFT JOIN events conflicting ON conflicting.stay_id=e.stay_id
            AND conflicting.variable=e.variable
            AND conflicting.measured_at=e.measured_at
            AND conflicting.event_time<=q.t AND conflicting.value<>e.value
        WHERE conflicting.source_id IS NULL
        ), ranked AS (
        SELECT *,dense_rank() OVER(PARTITION BY query_id ORDER BY measured_at DESC) AS draw_rank
        FROM candidates)
        SELECT query_id,min(value) AS value FROM ranked WHERE draw_rank=1
        GROUP BY query_id HAVING count(distinct value)=1
            AND bool_and(coalesce(quality_valid,false))
            AND bool_and(t-measured_at<=INTERVAL '1 hour')
        ORDER BY query_id""").fetchdf()
    db.close()
    values=np.full(len(q),np.nan)
    values[result.query_id.to_numpy(dtype=int)]=result.value.to_numpy(float)
    return values


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    out=Path('artifacts/native_reconstruction_additional_domain_scaling_20260922.json')
    private=root/'scaling_20260922.pt'
    if out.exists() or private.exists():raise ValueError('Existing scaling')
    pr=Path('artifacts/native_reconstruction_additional_domain_partitions_20260922.json')
    report=json.loads(pr.read_text());fp=root/'partitions_20260922.pt'
    if sha256(fp)!=report['private_sha256']:raise ValueError('Changed roles')
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed partition inputs')
    plan=torch.load(fp,weights_only=False)
    indices={};paths=[Path(__file__),pr,fp,Path('scripts/encoder_fold_scaling.py')]
    for kind in ('labs','bedside'):
        ip=root/kind/'event_index.pt';paths.append(ip)
        indices[kind]={s['hospitalid']:s for s in torch.load(ip,weights_only=False)}
    mp=root/'masked_history_full/index.pt';paths.append(mp)
    masked=torch.load(mp,weights_only=False)
    labels=[]
    for item in masked:
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Changed decisions')
        q=pd.read_parquet(item['path']);q=q.loc[q.status.eq('accepted')].reset_index(drop=True)
        if q.empty:continue
        pieces=[]
        for kind in indices:
            source=indices[kind][item['hospitalid']]
            if sha256(Path(source['path']))!=source['sha256']:raise ValueError('Changed raw events')
            pieces.append(pd.read_parquet(source['path']))
        q['value']=anchor_labels(pd.concat(pieces,ignore_index=True),q)
        if not np.isfinite(q.value).all():raise ValueError('Accepted label fails exact value extraction')
        labels.append(q[['uniquepid','variable','value']])
    values=pd.concat(labels,ignore_index=True)
    if values.duplicated(['uniquepid','variable']).any():raise ValueError('Duplicate patient center')
    cp=Path('models/whole_body_joint_jepa.pt');paths.append(cp)
    variables=torch.load(cp,weights_only=False)['variables']
    people=sorted(plan['folds'][0])
    frame=values.pivot(index='uniquepid',columns='variable',values='value').reindex(index=people,columns=variables)
    centers=frame.to_numpy(float);scalers=[];summary=[]
    for number,roles in enumerate(plan['folds']):
        fit=np.array([roles[p]=='fit' for p in people]);scaler=fit_scaler(centers,fit)
        # Deliberately corrupt every held-out center to verify no fitted dependence.
        altered=centers.copy();altered[~fit]=1e12
        check=fit_scaler(altered,fit)
        for k in scaler:np.testing.assert_array_equal(scaler[k],check[k])
        scalers.append(scaler)
        summary.append(dict(fold=number,fit_people=int(fit.sum()),
            variables={v:dict(fit_patients=int(scaler['counts'][i]),location=float(scaler['location'][i]),
                scale=float(scaler['scale'][i])) for i,v in enumerate(variables)},
            unsupported_variables=[v for i,v in enumerate(variables) if scaler['counts'][i]==0],
            heldout_mutation_invariant=True))
    bindings={str(p):sha256(p) for p in paths}
    torch.save(dict(variables=variables,people=people,centers=centers,scalers=scalers,bindings=bindings),private)
    result=dict(accepted_labels=len(values),people_with_labels=int(values.uniquepid.nunique()),
        folds=summary,bindings=bindings,private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        policy='One accepted fixed raw-unit label per patient-variable; fit-patient median/IQR only. Missing centers remain NaN. Zero IQR uses one canonical unit; missing support uses mask/abstention, not inferred normality.',
        limitation='Current-label scaling, not trained model efficacy. Unlike old sample-history summaries, no median fallback for variables without an accepted anchor; fit-only input vocabulary replay still required. Single-source-version restriction explicitly checked. No calibration/selection/test values fit scalers.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(accepted_labels=len(values),people_with_labels=int(values.uniquepid.nunique()),
        folds=[dict(fold=f['fold'],unsupported=f['unsupported_variables']) for f in summary]),indent=2),flush=True)


if __name__=='__main__':main()

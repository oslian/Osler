"""Retrospective record-triggered queries; not verified real-time availability."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/rate_anchored_potassium_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    oldroot=Path('/private/tmp/insulin_potassium_pairs_20260911')
    dest=Path('/private/tmp/rate_anchored_potassium_20260911');dest.mkdir(exist_ok=True)
    prior=json.loads(Path('artifacts/insulin_potassium_partition_20260911.json').read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(oldroot/name)!=prior['binding'][str(oldroot/name)]:raise ValueError('cache changed')
    oldlabels=np.load(oldroot/'labels.npz')
    oldq=pd.read_parquet(oldroot/'queries.parquet').loc[oldlabels['paired'][:,0]].reset_index(drop=True)
    rates=read_insulin_rates(Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz'))
    rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    rates=rates[rates.stay_id.isin(oldq.stay_id)]
    pieces=[]
    for (stay,label),g in rates.groupby(['stay_id','label']):
        conflicts=g.groupby('offset').value.nunique()
        g=g[~g.offset.isin(conflicts[conflicts>1].index)].sort_values('offset').drop_duplicates('offset')
        previous=g.offset.shift();delta=g.value.diff()
        valid=delta.abs().gt(1e-8)&(g.offset-previous).le(360)
        p=g.loc[valid,['stay_id','offset']].copy();p['previous_offset']=previous[valid]
        pieces.append(p)
    q=pd.concat(pieces,ignore_index=True)
    # Multiple simultaneous labels do not define a single attributable rate change.
    q=q[~q.duplicated(['stay_id','offset'],keep=False)].reset_index(drop=True)
    q['t']=pd.Timestamp('2100-01-01')+pd.to_timedelta(q.offset,unit='m')
    meta=oldq[['stay_id','uniquepid','hospitalid']].drop_duplicates()
    q=q.merge(meta,on='stay_id',validate='many_to_one')
    raw=pd.read_parquet(oldroot/'labs.parquet')
    labels=future_pairs(q,raw,['potassium'],horizon=6,tolerance=1,max_age=6)
    valid=labels['paired'][:,0]
    draw=q.offset.to_numpy()-labels['measurement_age'][:,0]*60
    after=valid&(q.previous_offset.to_numpy()>=draw)
    def stats(mask):
        return dict(rows=int(mask.sum()),patients=int(q.loc[mask,'uniquepid'].nunique()),hospitals=int(q.loc[mask,'hospitalid'].nunique()))
    report=dict(complete=True,status='record_anchored_support_not_accuracy',runtime_modified=False,treatment_availability_verified=False,
                all=stats(np.ones(len(q),bool)),paired=stats(valid),bracket_after_draw=stats(after),seeds=[])
    for entry in prior['seeds']:
        path=oldroot/f"partition_{entry['seed']}.npz"
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        oldroles=np.load(path)
        roles={k:q.uniquepid.isin(set(oldq.loc[oldroles[k],'uniquepid'])).to_numpy() for k in ('fit','tune','evaluation','hospital')}
        # Keep hospital role restricted to the same held-out hospitals.
        held=set(oldq.loc[oldroles['hospital'],'hospitalid'])
        roles['hospital'] = roles['hospital'] & q.hospitalid.isin(held).to_numpy()
        p=dest/f"partition_{entry['seed']}.npz";np.savez_compressed(p,**roles)
        report['seeds'].append(dict(seed=entry['seed'],roles={k:dict(paired=stats(m&valid),bracket_after=stats(m&after)) for k,m in roles.items()},partition_sha256=sha256(p)))
    q.to_parquet(dest/'queries.parquet',index=False)
    np.savez_compressed(dest/'labels.npz',**labels,bracket_after=after)
    report['binding']={str(p):sha256(p) for p in [Path(__file__),oldroot/'labs.parquet',dest/'queries.parquet',dest/'labels.npz']}
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

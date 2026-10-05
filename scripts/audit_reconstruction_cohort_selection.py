"""Quantify inherited insulin-rate cohort restriction, not treatment effects."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_cohort_selection_20260917.json')
    if out.exists():raise ValueError('Existing audit')
    qp=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    newp=Path('data/research_recovery/expanded_confirmation_20260917/provenance.parquet')
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    raw=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    q=pd.read_parquet(qp,columns=['stay_id','uniquepid']).drop_duplicates()
    new=pd.read_parquet(newp,columns=['stay_id','uniquepid']).drop_duplicates()
    roles=torch.load(sp,weights_only=False)['roles']
    rates=read_insulin_rates(raw);rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    changing=[]
    for (stay,label),g in rates.groupby(['stay_id','label']):
        conflicts=g.groupby('offset').value.nunique()
        g=g[~g.offset.isin(conflicts[conflicts>1].index)].sort_values('offset').drop_duplicates('offset')
        valid=g.value.diff().abs().gt(1e-8)&g.offset.diff().le(360)
        changing.extend((stay,float(o)) for o in g.loc[valid,'offset'])
    changes=pd.DataFrame(changing,columns=['stay_id','offset'])
    changes=changes[~changes.duplicated(['stay_id','offset'],keep=False)]
    any_rate=set(rates.stay_id);changed=set(changes.stay_id)
    def describe(frame):
        return dict(stays=len(frame),patients=int(frame.uniquepid.nunique()),
            stays_with_any_insulin_rate=int(frame.stay_id.isin(any_rate).sum()),
            stays_with_unambiguous_rate_change=int(frame.stay_id.isin(changed).sum()),
            patients_with_rate_change=int(frame.loc[frame.stay_id.isin(changed),'uniquepid'].nunique()))
    cohorts={r:describe(q.loc[q.uniquepid.map(roles)==r]) for r in sorted(set(roles.values()))}
    cohorts['expanded_confirmation']=describe(new)
    if cohorts['readout_tune']['stays_with_unambiguous_rate_change']!=cohorts['readout_tune']['stays']:
        raise ValueError('Expected historical rate restriction did not reproduce')
    cache_path=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    old=torch.load(cache_path,weights_only=False)
    index_path=Path('data/research_recovery/expanded_confirmation_20260917/predictions/index.pt')
    index=torch.load(index_path,weights_only=False);rows=[]
    for entry in index:
        p=Path(entry['path'])
        if sha256(p)!=entry['sha256']:raise ValueError('Changed predictions')
        rows.extend(torch.load(p,weights_only=False))
    stats={}
    for target in sorted({d['target'] for d in rows}):
        before=[d for d in old if d['target']==target and roles[d['subject_id']]=='readout_tune']
        after=[d for d in rows if d['target']==target]
        def summary(part):
            y=np.array([d['truth'] for d in part]);original=np.array([d['original'] for d in part])
            return dict(patients=len(y),truth_quantiles=np.quantile(y,[.1,.5,.9]).tolist(),original_mae=float(abs(original-y).mean()))
        stats[target]=dict(original_confirmation=summary(before),expanded_confirmation=summary(after))
    paths=[Path(__file__),qp,newp,sp,raw,cache_path,index_path,Path('scripts/build_rate_anchored_potassium.py'),
           Path('scripts/audit_map_native_anchor_support.py'),Path('scripts/build_native_reconstruction_expanded.py'),
           Path('scripts/audit_eicu_insulin_sequence_inventory.py'),Path('whole_body_jepa/event_treatment.py')]
    result=dict(complete=True,cohorts=cohorts,targets=stats,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Retrospective entire-stay rate support diagnoses cohort selection only; not pre-anchor feature, evidence of drug effect, or permission to gate predictions. Quantiles normalized; simultaneous extraction and sampling differences remain possible. No causal root-cause isolation.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(cohorts,indent=2))


if __name__=='__main__':main()

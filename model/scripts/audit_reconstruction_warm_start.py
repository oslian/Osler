"""Verify candidate initialization ancestry and cohort overlap, aggregate only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from whole_body_joint_jepa import _event_module_variable_contracts
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_warm_start_20260917.json')
    if out.exists():raise ValueError('existing audit')
    cp=Path('models/whole_body_joint_jepa.pt')
    original=Path('/Users/chunyouchang/Desktop/llm_project/medical_jepa/whole_body_joint_jepa_full_canonical_5000_candidate_20260808.pt')
    rp=original.with_suffix('.json')
    if sha256(cp)!=sha256(original):raise ValueError('ancestry checkpoint mismatch')
    ck=torch.load(cp,weights_only=False);report=json.loads(rp.read_text())
    source=Path(ck['training']['event_examples'])
    schema=pq.read_schema(source).names
    variables=ck['variables']
    cols=['subject_id','stay_id','hospitalid','_source_horizon']+[f'_observed_{v}' for v in variables]
    if not set(cols).issubset(schema):raise ValueError('source observation contract missing')
    frame=pd.read_parquet(source,columns=cols)
    frame=frame.loc[frame._source_horizon.isin([1,3,6,12,24,48])].copy()
    if frame.stay_id.nunique()>5000:raise ValueError('sampling requires exact historical selection')
    groups=_event_module_variable_contracts(report['modules'],variables,contract_mode=report['organ_contract_mode'])
    count=np.zeros(len(frame),int)
    for vs in groups.values():
        count+=(frame[[f'_observed_{v}' for v in vs]].fillna(0).to_numpy(float)>.5).any(1)
    frame=frame.loc[count>=ck['training']['joint_rows_min_modules']]
    reproduced = len(frame)==ck['training']['rows'] and frame.subject_id.nunique()==ck['training']['subjects']
    patients=set(frame.subject_id.astype(str))
    split=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    roles=torch.load(split,weights_only=False)['roles']
    overlaps={r:dict(patients=sum(role==r for role in roles.values()),
        in_current_reconstructed_source_cohort=sum(str(p) in patients for p,role in roles.items() if role==r)) for r in sorted(set(roles.values()))}
    paths=[Path(__file__),cp,original,rp,source,split,Path('whole_body_joint_jepa.py')]
    result=dict(complete=True,checkpoint_identical=True,historical_cohort_reproduced=reproduced,
        historical_rows=ck['training']['rows'],historical_subjects=ck['training']['subjects'],
        current_reconstructed_rows=len(frame),current_reconstructed_subjects=len(patients),
        independence_certified=False,
        roles=overlaps,bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Checkpoint byte ancestry verified, but historical source hash/optimizer batch membership not stored. Current source reconstruction is NOT certified historical training membership, especially when aggregate counts differ. No exact patient exposure claim or blanket invalidation of separate external release audits.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('cohort reproduced',reproduced,'current',len(frame),len(patients),'historical',ck['training']['rows'],ck['training']['subjects'],overlaps,flush=True)


if __name__=='__main__':main()

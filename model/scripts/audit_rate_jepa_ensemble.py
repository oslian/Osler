"""Fixed equal-weight ensemble on common held-out hospital patients only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/rate_jepa_ensemble_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    ref=json.loads(Path('artifacts/rate_jepa_hospitals_20260911.json').read_text())
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    points={k:[] for k in ('own','whole')};hospital=None
    for item in ref['seeds']:
        seed=item['seed'];p=root/f'jepa_predictions_{seed}.npz'
        if sha256(p)!=ref['binding'][str(p)]:raise ValueError('predictions changed')
        part={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        if hospital is not None:np.testing.assert_array_equal(hospital,part['hospital'])
        hospital=part['hospital']
        if set(q.loc[hospital,'subject_id']) & set(q.loc[part['fit']|part['tune'],'subject_id']):raise ValueError('ensemble patient leakage')
        saved=np.load(p)
        for k in points:points[k].append(saved[k])
    means={k:np.mean(v,axis=0) for k,v in points.items()}
    means['persistence']=labels['current'][valid,0]
    report=dict(complete=True,runtime_modified=False,fitting_performed=False,
                policy='fixed_seven_seed_equal_mean',validation_only=True,groups={})
    for name,g in [('all',np.ones(len(q),bool)),('bracket_after',labels['bracket_after'][valid])]:
        mask=hospital&g;f=q.loc[mask]
        comparisons={b:paired(f,means['whole'][mask],means[b][mask],f'ensemble-{name}-{b}',1000) for b in ('own','persistence')}
        report['groups'][name]=dict(patients=int(f.subject_id.nunique()),mae={k:patient_equal_mae(f,f.truth,p[mask]) for k,p in means.items()},
            comparisons=comparisons,versus_individual_whole=[paired(f,means['whole'][mask],p[mask],f'ensemble-{name}-{i}',1000) for i,p in enumerate(points['whole'])])
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

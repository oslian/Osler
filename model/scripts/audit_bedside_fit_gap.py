"""Locate numeric increment loss before versus after fitting-patient evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_bedside_fit_gap_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    parents={mode:json.loads(Path(f'artifacts/native_{tag}_training_20260911.json').read_text())
        for mode,tag in [('real','bedside_rescale'),('support','bedside_support')]}
    valid=np.load(root/'labels.npz')['paired'][:,0]
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));col=labels['variables'].tolist().index('glucose')
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];items=[];bindings={}
    for seed in (7,19,31,43,59,71,83):
        points={}
        for mode,tag in [('real','bedside_rescale'),('support','bedside_support')]:
            path=root/f'{tag}_predictions_{seed}.npz'
            if sha256(path)!=parents[mode]['bindings'][str(path)]:raise ValueError('predictions changed')
            bindings[str(path)]=sha256(path)
            data=np.load(path);points[mode]=data['bedside']
            if 'lab' in points:np.testing.assert_array_equal(points['lab'],data['lab'])
            points['lab']=data['lab']
        path=root/f'partition_{seed}.npz'
        if sha256(path)!=parents['real']['bindings'][str(path)]:raise ValueError('partition changed')
        bindings[str(path)]=sha256(path)
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(path)).items()}
        item=dict(seed=seed,roles={})
        for role in ('fit','tune','evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(patients=int(f.subject_id.nunique()),
                mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'bedsidefit-{seed}-{role}-{a}-{b}',1000)
                    for a,b in [('real','support'),('real','lab'),('support','lab')]})
        items.append(item)
    for path in (Path(__file__),root/'queries.parquet',root/'labels.npz',root/'multitarget_labels_joint_conflict_checked.npz'):
        bindings[str(path)]=sha256(path)
    output.write_text(json.dumps(dict(complete=True,training_performed=False,runtime_modified=False,
        seeds=items,bindings=bindings),indent=2)+'\n')
    for role in ('fit','tune','evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':main()

"""Whole measured-state nearest-patient support, source-defined diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    out=Path('artifacts/clean_state_support_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['binding'].items():
        if sha256(Path(p))!=h:raise ValueError('parent changed')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    base=dict(np.load(root/'body_arrays_conflict_checked.npz'));rows=base['query_rows']
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));col=list(labels['variables']).index('glucose')
    q['truth']=labels['future'][:,col]
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        scope='source-defined geometric support, not biological equivalence or routing',
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        fit=roles['fit']
        ck=np.load(root/f'conflict_repair_predictions_{seed}.npz')
        supported=base['mask'][rows[fit]].any(axis=0)
        x=np.column_stack([base[k][rows][:,supported] for k in ('current','mask','ages')]).astype(float)
        f=pd.DataFrame(x[fit]);f['subject']=q.loc[fit,'subject_id'].to_numpy()
        representatives=f.groupby('subject').median()
        center=representatives.median().to_numpy();scale=representatives.std().fillna(0).to_numpy()
        scale=np.where(scale>1e-6,scale,1.)
        reps=(representatives.to_numpy()-center)/scale;z=(x-center)/scale
        distances=cdist(reps,reps);np.fill_diagonal(distances,np.inf)
        threshold=float(np.quantile(distances.min(axis=1),.90))
        nearest=cdist(z,reps).min(axis=1)
        points={}
        for role in ('evaluation','hospital'):
            points[role]={}
            for name,condition in [('near',nearest<=threshold),('far',nearest>threshold)]:
                m=roles[role]&condition;fr=q.loc[m]
                points[role][name]=dict(rows=int(m.sum()),patients=int(fr.subject_id.nunique()),
                    comparison=paired(fr,ck['clean_full'][m],ck['clean_own'][m],f'geometry-{seed}-{role}-{name}',1000))
        result['seeds'].append(dict(seed=seed,threshold=threshold,roles=points))
    out.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for name in ('near','far'):
            gs=[s['roles'][role][name] for s in result['seeds']]
            print(role,name,'patients',[g['patients'] for g in gs],
                'gain',sum(g['comparison'].get('supported',False) and g['comparison']['ci95'][1]<0 for g in gs),
                'harm',sum(g['comparison'].get('supported',False) and g['comparison']['ci95'][0]>0 for g in gs))


if __name__=='__main__':main()

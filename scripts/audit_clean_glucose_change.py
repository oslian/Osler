"""Source-defined change-size diagnostic, not prospective regime routing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    out=Path('artifacts/clean_glucose_change_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['binding'].items():
        if sha256(Path(p))!=h:raise ValueError('parent changed')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));col=list(labels['variables']).index('glucose')
    q['truth']=labels['future'][:,col];delta=q.truth.to_numpy()-labels['current'][:,col]
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        scope='future-defined strata descriptive only; no clinical thresholds or prospective selector',
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        fit=roles['fit'];absfit=np.abs(delta[fit]);subjects=q.loc[fit,'subject_id']
        weights=1/subjects.map(subjects.value_counts()).to_numpy(float)
        order=np.argsort(absfit);cdf=np.cumsum(weights[order])/weights.sum()
        cutoff=float(absfit[order][np.searchsorted(cdf,.75)])
        pred=dict(np.load(root/f'conflict_repair_predictions_{seed}.npz'));item=dict(seed=seed,cutoff=cutoff,roles={})
        for role in ('fit','evaluation','hospital'):
            r=roles[role];item['roles'][role]={}
            for name,condition in [('large',np.abs(delta)>cutoff),('smaller',np.abs(delta)<=cutoff)]:
                m=r&condition;f=q.loc[m]
                stats={}
                for arm in ('clean_own','clean_full'):
                    d=pred[arm][m]-labels['current'][m,col]
                    tmp=pd.DataFrame({'subject':f.subject_id.to_numpy(),'absolute_delta':np.abs(delta[m]),
                        'absolute_prediction':np.abs(d),'direction':(np.sign(d)==np.sign(delta[m])).astype(float)})
                    stats[arm]=tmp.groupby('subject').mean().mean().to_dict()
                item['roles'][role][name]=dict(rows=int(m.sum()),patients=int(f.subject_id.nunique()),statistics=stats,
                    comparison=paired(f,pred['clean_full'][m],pred['clean_own'][m],f'clean-change-{seed}-{role}-{name}',1000))
        result['seeds'].append(item)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('cutoffs',[s['cutoff'] for s in result['seeds']])
    for role in ('fit','evaluation','hospital'):
        for name in ('large','smaller'):
            rs=[s['roles'][role][name] for s in result['seeds']]
            print(role,name,'patients',[r['patients'] for r in rs],
                'gain',sum(r['comparison']['ci95'][1]<0 for r in rs),
                'harm',sum(r['comparison']['ci95'][0]>0 for r in rs),
                'delta',float(np.median([r['statistics']['clean_full']['absolute_delta'] for r in rs])),
                'pred_delta',float(np.median([r['statistics']['clean_full']['absolute_prediction'] for r in rs])))


if __name__=='__main__':main()

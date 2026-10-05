"""Equalize repeated outcome events before equalizing patient errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_magnesium_anchor_overlap import describe
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    out=Path('artifacts/clean_future_event_weights_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['binding'].items():
        if sha256(Path(p))!=h:raise ValueError('parent changed')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    col=list(labels['variables']).index('glucose')
    q['future_id']=labels['future_source_id'][:,col];q['truth']=labels['future'][:,col]
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        scope='sensitivity audit, not replacement promotion gate',
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        pred=dict(np.load(root/f'conflict_repair_predictions_{seed}.npz'))
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m,['stay_id','uniquepid','future_id','truth']].copy()
            for a,p in pred.items():f[a]=np.abs(p[m]-f.truth)
            group=f.groupby(['stay_id','uniquepid','future_id'])
            if (group.truth.nunique()>1).any():raise ValueError('same outcome inconsistent truth')
            event=group[list(pred)].mean().reset_index()
            patient=event.groupby('uniquepid')[list(pred)].mean()
            item['roles'][role]=dict(rows=int(m.sum()),events=len(event),patients=len(patient),mae=patient.mean().to_dict(),
                comparisons={f'{a}_vs_{b}':describe(patient[a]-patient[b],seed)
                    for a,b in [('clean_full','clean_own'),('clean_full','old_full'),('clean_full','persistence')]})
        result['seeds'].append(item)
    out.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        rs=[s['roles'][role] for s in result['seeds']]
        print(role,'rows_events',[(r['rows'],r['events']) for r in rs],
              'mae',{a:float(np.median([r['mae'][a] for r in rs])) for a in pred},
              'comparisons',{a:{'gain':sum(r['comparisons'][a]['ci95'][1]<0 for r in rs),
                               'harm':sum(r['comparisons'][a]['ci95'][0]>0 for r in rs)} for a in rs[0]['comparisons']})


if __name__=='__main__':main()

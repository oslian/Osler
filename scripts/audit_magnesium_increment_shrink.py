"""Source-selected shrinkage of the matched whole-minus-own increment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=Path('artifacts/native_magnesium_matched_20260911.json')
    ref=json.loads(path.read_text())
    for name,digest in ref['binding'].items():
        assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_increment_shrink_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    labels=dict(np.load(root/'multitarget_labels.npz'));variables=list(labels['variables'])
    col=variables.index('magnesium');valid=np.load(root/'labels.npz')['paired'][:,0]
    eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];points=dict(np.load(root/f'magnesium_matched_predictions_{seed}.npz'))
        roles={k:v[valid]&eligible for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        grid=np.linspace(0,1,11);tune=roles['tune'];f=q.loc[tune]
        delta=points['real']-points['own']
        scores=[patient_equal_mae(f,f.truth,points['own'][tune]+a*delta[tune]) for a in grid]
        alpha=float(grid[np.argmin(scores)])
        points['selected']=points['own']+alpha*delta
        item=dict(seed=seed,alpha=alpha,tune_grid_mae=scores,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={a:paired(f,points['selected'][m],points[a][m],f'mg-shrink-{seed}-{role}-{a}',2000) for a in ('own','real','persistence')})
        items.append(item)
    report=dict(complete=True,neural_training_performed=False,runtime_modified=False,alpha_grid=grid.tolist(),seeds=items,
        binding={str(p):sha256(p) for p in (path,Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('alpha',[s['alpha'] for s in items])
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in ('own','real','persistence')})


if __name__=='__main__':main()

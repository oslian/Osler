"""Numerical CI stability, not model tuning or promotion by resampling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/bedside_common_bootstrap_20260912_v2.json')
    if out.exists():raise ValueError('fresh output required')
    parent=Path('artifacts/native_bedside_fit_gap_20260912.json')
    for path,digest in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(path))!=digest:raise ValueError('bound predictions or labels changed')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));col=labels['variables'].tolist().index('glucose')
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];items=[]
    for seed in (7,19,31,43,59,71,83):
        real=np.load(root/f'bedside_rescale_predictions_{seed}.npz');support=np.load(root/f'bedside_support_predictions_{seed}.npz')
        np.testing.assert_array_equal(real['lab'],support['lab'])
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            points=dict(real=real['bedside'][m],support=support['bedside'][m],lab=real['lab'][m])
            item['roles'][role]={str(draw_seed):{ref:paired_model_errors(f,points,ref,seed=draw_seed,repetitions=20000)
                for ref in ('lab','support')} for draw_seed in (20260912,20260913,20260914)}
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        bootstrap_draws=20000,seeds=items,bindings={str(p):sha256(p) for p in
            [parent,Path(__file__),Path('whole_body_jepa/change_evaluation.py')]}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for ref,model in [('lab','real'),('lab','support'),('support','real')]:
            print(role,model,ref,[(d,sum(s['roles'][role][str(d)][ref][model]['ci95'][1]<0 for s in items),
                sum(s['roles'][role][str(d)][ref][model]['ci95'][0]>0 for s in items)) for d in (20260912,20260913,20260914)])


if __name__=='__main__':main()

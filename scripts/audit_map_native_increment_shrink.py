"""Source-tune shrinkage of full-minus-own prediction; never test-selected."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_anchor_tree_20260912.json')
    out=Path('artifacts/map_native_increment_shrink_20260912.json')
    if out.exists():raise ValueError('fresh outputs required')
    ref=json.loads(parent.read_text())
    for p,h in ref['bindings'].items():assert sha256(Path(p))==h,p
    labels=dict(np.load(root/'labels.npz'));valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    grid=np.linspace(0,1,11);results=[]
    for entry in ref['seeds']:
        seed=entry['seed']
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        p=dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        tune=roles['tune'];delta=p['full']-p['own']
        scores=[patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p['own'][tune]+a*delta[tune]) for a in grid]
        alpha=float(grid[np.argmin(scores)])
        p['selected']=p['own']+alpha*delta
        item=dict(seed=seed,alpha=alpha,tune_scores=scores,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m];points={k:v[m] for k,v in p.items()}
            item['roles'][role]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in points.items()},
                comparisons={k:paired_model_errors(f,points,k,seed=20260912)['selected'] for k in ('own','full','persistence')})
        results.append(item)
    out.write_text(json.dumps(dict(complete=True,grid=grid.tolist(),seeds=results,
        neural_training_performed=False,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('whole_body_jepa/change_evaluation.py'))}),indent=2)+'\n')
    print('alpha',[s['alpha'] for s in results])
    for role in ('evaluation','hospital'):
        print(role,{k:float(np.median([r['roles'][role]['mae'][k] for r in results])) for k in p})
        for k in ('own','full','persistence'):
            c=[r['roles'][role]['comparisons'][k] for r in results]
            print('selected vs',k,'gain/harm',sum(v['ci95'][1]<0 for v in c),sum(v['ci95'][0]>0 for v in c))


if __name__=='__main__':main()

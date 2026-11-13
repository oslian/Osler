"""Matched shrinkage control isolates numerical versus support increments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_increment_shrink_20260912.json')
    training=Path('artifacts/map_native_anchor_tree_20260912.json')
    output=Path('artifacts/map_native_shrink_support_20260912.json')
    if output.exists():raise ValueError('fresh outputs required')
    for path in (parent,training):
        for p,h in json.loads(path.read_text())['bindings'].items():assert sha256(Path(p))==h,p
    prior=json.loads(parent.read_text())
    labels=dict(np.load(root/'labels.npz'));valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    grid=np.asarray(prior['grid']);results=[]
    for entry in prior['seeds']:
        seed=entry['seed'];p=dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        tune=roles['tune'];item=dict(seed=seed,selection={},roles={})
        for mode in ('full','support'):
            increment=p[mode]-p['own']
            scores=[patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p['own'][tune]+a*increment[tune]) for a in grid]
            alpha=float(grid[np.argmin(scores)])
            p['shrunk_'+mode]=p['own']+alpha*increment
            item['selection'][mode]=dict(alpha=alpha,tune_scores=scores)
            if mode=='full':
                assert alpha==entry['alpha']
                np.testing.assert_array_equal(scores,entry['tune_scores'])
        for role in ('evaluation','hospital'):
            m=roles[role];frame=q.loc[m];points={k:v[m] for k,v in p.items()}
            item['roles'][role]=dict(mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in points.items()},
                versus_shrunk_support=paired_model_errors(frame,points,'shrunk_support',seed=20260912),
                versus_own=paired_model_errors(frame,points,'own',seed=20260912))
            np.testing.assert_allclose(item['roles'][role]['mae']['shrunk_full'],entry['roles'][role]['mae']['selected'],rtol=1e-12)
        results.append(item)
    output.write_text(json.dumps(dict(complete=True,seeds=results,original_full_reproduced=True,
        runtime_modified=False,training_performed=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (parent,training,Path(__file__),Path('whole_body_jepa/change_evaluation.py'))}),indent=2)+'\n')
    print('support alpha',[r['selection']['support']['alpha'] for r in results])
    for role in ('evaluation','hospital'):
        print(role,{k:float(np.median([r['roles'][role]['mae'][k] for r in results])) for k in ('own','shrunk_full','shrunk_support')})
        for refname,arm in (('shrunk_support','shrunk_full'),('own','shrunk_support')):
            c=[r['roles'][role]['versus_'+refname][arm] for r in results]
            print(arm,'vs',refname,'gain/harm',sum(v['ci95'][1]<0 for v in c),sum(v['ci95'][0]>0 for v in c))


if __name__=='__main__':main()

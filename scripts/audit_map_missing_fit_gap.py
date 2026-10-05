"""Locate numerical increment in the completed native-missing comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_missing_support_20260912.json')
    output=Path('artifacts/map_missing_fit_gap_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['bindings'].items():
        assert sha256(Path(p))==h,p
    labels=dict(np.load(root/'labels.npz'))
    valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q=q.rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    results=[]
    for entry in ref['seeds']:
        seed=entry['seed']
        points=dict(np.load(root/f'tree_missing_support/predictions_{seed}.npz'))
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        scores={}
        for role,m in roles.items():
            f=q.loc[m]
            p={k:v[m] for k,v in points.items()}
            scores[role]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},
                comparisons={k:paired_model_errors(f,p,k,seed=20260912) for k in ('own','support')})
            if role in entry['roles']:
                assert scores[role]==entry['roles'][role]
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        seeds=results,bindings={str(p):sha256(p) for p in (parent,Path(__file__))}),indent=2)+'\n')
    for role in roles:
        rs=[r['roles'][role] for r in results]
        print(role,{k:float(np.median([r['mae'][k] for r in rs])) for k in ('own','full','support')})
        for b in ('own','support'):
            s=[r['comparisons'][b]['full'] for r in rs]
            print('full vs',b,'mean',float(np.median([v['mean'] for v in s])),
                'gain/harm',sum(v['ci95'][1]<0 for v in s),sum(v['ci95'][0]>0 for v in s))


if __name__=='__main__':
    main()

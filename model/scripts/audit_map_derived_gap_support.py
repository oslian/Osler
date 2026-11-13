"""Pre-anchor support strata for frozen derived-gap predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def support_groups(d):
    current=d[:,1,-1]>0
    any_history=(d[:,1]>0).any(axis=1)
    return dict(current=current,history_only=any_history&~current,none=~any_history)


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_derived_gap_training_20260912.json')
    output=Path('artifacts/map_derived_gap_support_strata_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['bindings'].items():
        assert sha256(Path(p))==h,p
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q=q.rename(columns={'uniquepid':'subject_id'})
    q['truth']=np.load(root/'labels.npz')['future'][valid,0]
    groups=support_groups(np.load(root/'tree_derived_gap_trial/derived_features.npy'))
    assert (np.stack(list(groups.values())).sum(axis=0)==1).all()
    results=[]
    for entry in ref['seeds']:
        seed=entry['seed']
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        points=dict(np.load(root/f'tree_derived_gap_trial/predictions_{seed}.npz'))
        scores={}
        for role in ('evaluation','hospital'):
            m=roles[role]
            for k,mae in entry['roles'][role]['mae'].items():
                np.testing.assert_allclose(patient_equal_mae(q.loc[m],q.truth[m],points[k][m]),mae,rtol=1e-12)
            scores[role]={}
            for name,g in groups.items():
                keep=m&g
                f=q.loc[keep]
                p={k:v[keep] for k,v in points.items()}
                scores[role][name]=dict(rows=int(keep.sum()),patients=int(f.subject_id.nunique()),
                    comparisons={k:paired_model_errors(f,p,k,seed=20260912) for k in ('full','own','no_gap','derived_support')})
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        grouping_uses_future=False,seeds=results,bindings={str(p):sha256(p) for p in (parent,Path(__file__))}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in groups:
            rs=[r['roles'][role][group] for r in results]
            print(role,group,'patients',min(r['patients'] for r in rs),max(r['patients'] for r in rs))
            for b in ('full','own','no_gap','derived_support'):
                stats=[r['comparisons'][b]['derived'] for r in rs]
                print(b,'mean',float(np.median([s['mean'] for s in stats])),'gain/harm',
                    sum(s['ci95'] is not None and s['ci95'][1]<0 for s in stats),
                    sum(s['ci95'] is not None and s['ci95'][0]>0 for s in stats))


if __name__=='__main__':
    main()

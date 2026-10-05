"""Inspect actual candidate input coordinates without selecting external fixes."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_bedside_scale_20260911.json')
    if output.exists():raise ValueError('fresh report required')
    base=dict(np.load(root/'body_arrays_bedside.npz'))
    native=dict(np.load(root/'native_tokens_bedside.npz'))
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    variables=labels['variables'].tolist();n=len(variables)
    eligible=labels['paired'][:,variables.index('glucose')]
    valid=np.load(root/'labels.npz')['paired'][:,0]
    entries=[]
    for seed in (7,19,31,43,59,71,83):
        partition=dict(np.load(root/f'partition_{seed}.npz'))
        item=dict(seed=seed,roles={})
        for role in ('fit','evaluation','hospital'):
            selected=np.flatnonzero(partition[role][valid]&eligible)
            history=np.unique(np.concatenate([native['history_indices'][native['query_ptr'][i]:native['query_ptr'][i+1]] for i in selected]))
            result={}
            for v in ('heart_rate','map','o2sat','respiratory_rate','temperature'):
                i=variables.index(v);m=native['input'][history,n+i]>0
                z=native['input'][history[m],i]
                result[v]=dict(observed_tokens=len(z),q01_q50_q99=np.quantile(z,[.01,.5,.99]).tolist(),
                    abs_gt_5=float(np.mean(abs(z)>5)),abs_gt_10=float(np.mean(abs(z)>10)))
            item['roles'][role]=result
        entries.append(item)
    report=dict(complete=True,training_performed=False,runtime_modified=False,seeds=entries,
        bindings={str(p):sha256(p) for p in [root/'body_arrays_bedside.npz',root/'native_tokens_bedside.npz',
            root/'multitarget_labels_joint_conflict_checked.npz',Path(__file__)]})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(entries[0],indent=2))


if __name__=='__main__':main()

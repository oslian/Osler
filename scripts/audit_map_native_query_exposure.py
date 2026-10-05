"""Replay native-query training exposure before attributing the fit gap."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.audit_map_expanded_exposure import replay, summarize
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    output=Path('artifacts/map_native_query_exposure_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    lab=dict(np.load(root/'labels.npz'));q['truth']=lab['future'][:,0]
    items=[]
    for seed in (7,19,31,43,59,71,83):
        record=json.loads(Path(f'artifacts/map_native_query_training_{seed}_20260912.json').read_text())
        roles=dict(np.load(root/f'partition_{seed}.npz'))
        fit=roles['fit']&lab['paired'][:,0]
        pools=[np.asarray(g.index) for _,g in q.loc[fit].groupby('subject_id',sort=True)]
        snapshots=replay(pools,seed)
        seen={};details={}
        for arm in ('own','full'):
            epoch=record['arms'][arm]['selected_epoch']
            counts=snapshots[epoch-1] if epoch else {}
            mask=np.zeros(len(q),bool);mask[list(counts)]=True
            if (mask&~fit).any():raise ValueError('sampled nonfit row')
            seen[arm]=mask
            details[arm]=dict(selected_epoch=epoch,**summarize(counts,pools))
        groups={'both_seen':fit&seen['own']&seen['full'],
                'one_seen':fit&(seen['own']^seen['full']),
                'neither_seen':fit&~seen['own']&~seen['full']}
        np.testing.assert_array_equal(sum(g.astype(int) for g in groups.values()),fit.astype(int))
        pred=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        scored={}
        for name,m in groups.items():
            if not m.any():
                scored[name]=dict(rows=0);continue
            f=q.loc[m];p={k:a[m] for k,a in pred.items()}
            scored[name]=dict(rows=int(m.sum()),patients=f.subject_id.nunique(),
                mae={k:patient_equal_mae(f,f.truth,a) for k,a in p.items()},
                full_vs_own=paired_model_errors(f,p,'own',seed=20260912,repetitions=20000)['full'])
        items.append(dict(seed=seed,arms=details,epoch32=summarize(snapshots[-1],pools),groups=scored))
    summary={name:dict(
        median_paired_delta=float(np.median([x['groups'][name]['full_vs_own']['mean'] for x in items])),
        gain=sum(x['groups'][name]['full_vs_own']['ci95'][1]<0 for x in items),
        harm=sum(x['groups'][name]['full_vs_own']['ci95'][0]>0 for x in items))
        for name in ('both_seen','neither_seen')}
    result=dict(complete=True,items=items,summary=summary,runtime_modified=False,
        scope='deterministic sampler replay; selected fit predictions, not efficacy or a budget intervention',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_map_expanded_exposure.py'))})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(summary)
    print({'epoch32_row_coverage':[x['epoch32']['row_coverage'] for x in items],
           'selected_full_coverage':[x['arms']['full']['row_coverage'] for x in items]})


if __name__=='__main__':main()

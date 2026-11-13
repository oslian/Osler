"""Locate the earliest measured own/full gap using saved selected predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/map_native_query_fitgap_20260912.json')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    lab=dict(np.load(root/'labels.npz'));q['truth']=lab['future'][:,0]
    items=[]
    for seed in (7,19,31,43,59,71,83):
        predictions=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        roles=dict(np.load(root/f'partition_{seed}.npz'))
        original=json.loads(Path(f'artifacts/map_native_query_training_{seed}_20260912.json').read_text())
        item=dict(seed=seed,roles={})
        for role in ('fit','tune','evaluation','hospital'):
            m=roles[role]&lab['paired'][:,0];f=q.loc[m]
            p={k:a[m] for k,a in predictions.items()}
            mae={k:patient_equal_mae(f,f.truth,a) for k,a in p.items()}
            contrast=paired_model_errors(f,p,'own',seed=20260912,repetitions=20000)['full']
            if role in ('evaluation','hospital'):
                assert mae==original['roles'][role]['mae']
                assert contrast==original['roles'][role]['comparisons']['full_vs_own']
            if role=='tune':
                for arm in ('own','full'):
                    np.testing.assert_allclose(mae[arm],original['arms'][arm]['best_tune_mae'],rtol=0,atol=1e-10)
            item['roles'][role]=dict(mae=mae,full_vs_own=contrast)
        items.append(item)
    summary={role:dict(
        median_paired_delta=float(np.median([r['roles'][role]['full_vs_own']['mean'] for r in items])),
        gain=sum(r['roles'][role]['full_vs_own']['ci95'][1]<0 for r in items),
        harm=sum(r['roles'][role]['full_vs_own']['ci95'][0]>0 for r in items))
        for role in ('fit','tune','evaluation','hospital')}
    result=dict(complete=True,items=items,summary=summary,original_scores_exact=True,
        runtime_modified=False,scope='selected predictions; fit is resubstitution, not efficacy',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__))})
    output.write_text(json.dumps(result,indent=2)+'\n');print(summary)


if __name__=='__main__':main()

"""Fixed-model comparison by recorded PRE-anchor treatment change, not outcome."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/map_pressor_history_training_20260912.json')
    original = json.loads(parent.read_text())
    for p,h in original['bindings'].items(): assert sha256(Path(p)) == h
    past = np.load(root/'map_pressor_history.npz')
    q = pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels = np.load(root/'map_all_query_labels.npz'); q['truth'] = labels['future'][:,0]
    paired = np.zeros(len(q),bool); changed = paired.copy()
    for lag in (1,3):
        paired |= past[f'delta_known_lag{lag}'].any(axis=(1,2))
        d = past[f'delta_lag{lag}']; changed |= (np.isfinite(d)&(np.abs(d)>1e-8)).any(axis=(1,2))
    groups = {'recorded_change':changed, 'recorded_no_change':paired&~changed, 'no_comparable_pair':~paired}
    results = []
    for seed in (7,19,31,43,59,71,83):
        roles = np.load(root/f'partition_{seed}.npz')
        prediction = dict(np.load(root/f'map_pressor_history_training_predictions_{seed}.npz'))
        item = {'seed':seed,'roles':{}}
        for role in ('evaluation','hospital'):
            item['roles'][role] = {}
            for name, group in groups.items():
                m = roles[role]&labels['paired'][:,0]&group; frame = q.loc[m]
                p = {k:a[m] for k,a in prediction.items()}
                item['roles'][role][name] = dict(rows=int(m.sum()), patients=int(frame.subject_id.nunique()),
                    mae={k:patient_equal_mae(frame,frame.truth,a) for k,a in p.items()},
                    comparisons={r:paired_model_errors(frame,p,r,seed=20260912,repetitions=20000)['rate']
                                 for r in ('baseline','support')})
        results.append(item)
    paths=[Path(__file__),parent,root/'map_pressor_history.npz',root/'queries.parquet',root/'map_all_query_labels.npz']
    Path('artifacts/map_history_change_subgroups_20260912.json').write_text(json.dumps(dict(
        results=results,training_performed=False,runtime_modified=False,online_eligible=False,
        limitations=['Recorded-time only, not confirmed active treatment or availability.',
                     'Patients may occur in multiple strata; this is not trained prospective routing.',
                     'Pre-anchor dose change is distinct from future MAP change.'],
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for name in groups:
            ss=[i['roles'][role][name] for i in results]
            print(role,name,'patients',[s['patients'] for s in ss],
                {r: [(s['comparisons'][r]['mean'],s['comparisons'][r]['ci95']) for s in ss]
                 for r in ('baseline',)})


if __name__=='__main__':main()

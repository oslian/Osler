"""Pre-anchor sampling-gap strata, fixed predictions; no routing selection."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parents = [Path('artifacts/map_native_raw_density_20260912.json'),
               Path('artifacts/map_native_anchor_tree_20260912.json')]
    output = Path('artifacts/map_native_sampling_gap_20260912.json')
    if output.exists():
        raise ValueError('fresh outputs required')
    for parent in parents:
        for p,h in json.loads(parent.read_text())['bindings'].items():
            assert sha256(Path(p)) == h,p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][valid,0]
    density = dict(np.load(root/'raw_density.npz'))
    variables = joblib.load(root/'tree_trial/full_7.joblib')['variables']
    others = [variables.index(v) for v in ('heart_rate','o2sat','respiratory_rate','temperature','glucose')]
    gaps = (density['raw_count'][valid]>=2)&(density['snapshot_count'][valid]<2)
    othergap = gaps[:,others].any(axis=1)
    own_gap = gaps[:,variables.index('map')]
    groups = dict(other_gap=othergap, no_other_gap=~othergap,
                  own_gap=own_gap, no_own_gap=~own_gap)
    results = []
    for seed in (7,19,31,43,59,71,83):
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        points = dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        item = dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            item['roles'][role] = {}
            for name,g in groups.items():
                m = roles[role]&g
                frame = q.loc[m]
                p = {k:v[m] for k,v in points.items()}
                item['roles'][role][name] = dict(rows=int(m.sum()),patients=int(frame.subject_id.nunique()),
                    mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                    versus_own=paired_model_errors(frame,p,'own',seed=20260912))
        results.append(item)
    output.write_text(json.dumps(dict(complete=True,seeds=results,
        group_definition='At least two legal raw draw times but fewer than two snapshot draw proxies within6h',
        other_variables=['heart_rate','o2sat','respiratory_rate','temperature','glucose'],
        future_used_for_grouping=False,training_performed=False,runtime_modified=False,
        bindings={str(p):sha256(p) for p in parents+[Path(__file__),Path('whole_body_jepa/change_evaluation.py')]}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in groups:
            r=[s['roles'][role][group] for s in results]
            cs=[s['versus_own']['full'] for s in r]
            print(role,group,'patients',[s['patients'] for s in r],
                  'gain/harm',sum(c['ci95'] is not None and c['ci95'][1]<0 for c in cs),
                  sum(c['ci95'] is not None and c['ci95'][0]>0 for c in cs),
                  'mean_delta',float(np.median([c['mean'] for c in cs])))


if __name__ == '__main__':
    main()

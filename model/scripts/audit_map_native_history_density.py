"""Measure observed lag support, separating carried values from new draw times."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS


def draw_counts(mask, ages, lags, max_lookback_hours=None):
    """Second-rounded measured-time proxies, NOT raw source-event identities."""
    times = np.rint((-np.asarray(lags)[None, :, None]-ages*24.)*3600.)
    observed = mask > 0
    if max_lookback_hours is not None:
        observed &= times >= -max_lookback_hours*3600.
    times = np.where(observed, times, np.inf)
    times.sort(axis=1)
    first = np.isfinite(times[:, :1])
    new = np.isfinite(times[:, 1:]) & (times[:, 1:] != times[:, :-1])
    return np.concatenate([first,new], axis=1).sum(axis=1)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_anchor_inputs_20260912.json')
    output = Path('artifacts/map_native_history_density_v2_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(parent.read_text())
    q = pd.read_parquet(root/'queries.parquet')
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:,0]
    # Variable metadata is taken from the hash-bound fitted source model.
    import joblib
    modelpath = root/'tree_trial/full_7.joblib'
    trainingpath = Path('artifacts/map_native_anchor_tree_20260912.json')
    training = json.loads(trainingpath.read_text())
    assert sha256(modelpath) == training['bindings'][str(modelpath)]
    variables = joblib.load(modelpath)['variables']
    n = len(variables)
    current = np.zeros((len(q),n),bool)
    repeat = current.copy()
    counts = np.zeros((len(q),n),np.uint8)
    for shard in ref['shards']:
        assert sha256(Path(shard['path'])) == shard['sha256']
        a = dict(np.load(shard['path']))
        r = a['global_query_rows']
        h = a['history'][a['query_rows']]
        mask, age = a['mask'][h], a['ages'][h]
        current[r] = mask[:,-1]>0
        counts[r] = draw_counts(mask, age, HISTORY_LAGS_HOURS)
        recent = np.asarray(HISTORY_LAGS_HOURS) <= 6
        repeat[r] = draw_counts(mask[:,recent],age[:,recent],np.asarray(HISTORY_LAGS_HOURS)[recent],6.) >= 2
    roles = dict(np.load(root/'partition_7.npz'))
    summary = {}
    for role,m in roles.items():
        selected = m & valid
        def patient_average(a):
            return pd.DataFrame(a[selected]).groupby(q.loc[selected,'uniquepid'].to_numpy()).mean().mean().to_numpy()
        now, repeats, history = patient_average(current), patient_average(repeat), patient_average(counts>0)
        summary[role] = dict(rows=int(selected.sum()),patients=int(q.loc[selected,'uniquepid'].nunique()),
            variables=[dict(variable=v,current_fraction=float(now[j]),any_history_fraction=float(history[j]),
                            recent_repeated_draw_fraction=float(repeats[j])) for j,v in enumerate(variables)])
    report=dict(complete=True,roles=summary,training_performed=False,runtime_modified=False,
        caveat='Second-rounded measured-time proxies, not source IDs; same-time multiple tests merge. Seed7 descriptive only.',
        bindings={str(p):sha256(p) for p in (parent,trainingpath,modelpath,root/'queries.parquet',root/'labels.npz',
                                          root/'partition_7.npz',Path(__file__),Path('nonlinear_latent_rollout.py'))})
    output.write_text(json.dumps(report,indent=2)+'\n')
    for role,r in summary.items():
        print(role,'patients',r['patients'])
        for row in r['variables']:
            if row['any_history_fraction'] >= .2:
                print(row)


if __name__ == '__main__':
    main()

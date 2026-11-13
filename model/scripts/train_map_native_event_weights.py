"""Matched MAP training intervention: equal future events within each patient."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from scripts.train_map_native_anchor_tree import feature_view
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def event_weights(frame):
    keys = ['subject_id', 'stay_id', 'future_id']
    if frame.empty or frame[keys].isna().any().any() or (frame.future_id == '').any():
        raise ValueError('training identities must be complete')
    multiplicity = frame.groupby(keys).subject_id.transform('size').to_numpy(float)
    counts = frame[keys].drop_duplicates().groupby('subject_id').size()
    weights = 1 / (multiplicity * frame.subject_id.map(counts).to_numpy(float))
    return weights * len(weights) / weights.sum()


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_native_event_weight_training_20260912.json')
    dest = root/'tree_event_weight_trial'
    if output.exists() or dest.exists():
        raise ValueError('fresh outputs required')
    ref = json.loads(parent.read_text())
    for p, h in ref['bindings'].items():
        assert sha256(Path(p)) == h, p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'})
    q['future_id'] = labels['future_source_id'][valid, 0]
    q['truth'] = labels['future'][valid, 0]
    assert q.groupby(['subject_id','stay_id','future_id']).truth.nunique().max() == 1
    current = labels['current'][valid, 0]
    x = np.load(root/'tree_trial/features.npy', mmap_mode='r')
    assert len(x) == len(q)
    bindings = [parent, Path(__file__), Path('scripts/train_map_native_anchor_tree.py'),
                root/'tree_trial/features.npy']
    dest.mkdir()
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        fit = roles['fit']
        weights = event_weights(q.loc[fit])
        totals = pd.Series(weights).groupby(q.loc[fit].subject_id.to_numpy()).sum()
        np.testing.assert_allclose(totals, totals.iloc[0], rtol=1e-12)
        old = dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        points = dict(persistence=current, old_full=old['full'], old_own=old['own'])
        for mode in ('own','full','support'):
            pack = joblib.load(root/f'tree_trial/{mode}_{seed}.joblib')
            n, col = len(pack['variables']), pack['target_index']
            scale, allowed = pack['target_scale'], pack['allowed']
            model = clone(pack['model'])
            train = feature_view(x[fit], n, col, allowed, mode)
            model.fit(train, (q.truth.to_numpy()[fit]-current[fit])/scale, sample_weight=weights)
            del train
            pred = np.empty(len(q))
            for start in range(0,len(q),4096):
                end = min(len(q),start+4096)
                pred[start:end] = current[start:end]+scale*model.predict(feature_view(x[start:end],n,col,allowed,mode))
            points[mode] = pred
            pack['model'] = model
            pack['training_weight_policy'] = 'patient_then_future_event_equal'
            path = dest/f'{mode}_{seed}.joblib'
            joblib.dump(pack,path)
            bindings.append(path)
            print(f'Seed {seed} {mode} complete',flush=True)
        scores = {}
        for role in ('evaluation','hospital'):
            m = roles[role]
            frame = q.loc[m]
            p = {k:v[m] for k,v in points.items()}
            for name in ('own','full','support','persistence'):
                np.testing.assert_allclose(patient_equal_mae(frame,frame.truth,old[name][m]),entry['roles'][role]['mae'][name],rtol=1e-12)
            scores[role] = dict(patients=int(frame.subject_id.nunique()),
                mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                comparisons={name:paired_model_errors(frame,p,name,seed=20260912) for name in ('own','support','old_full','old_own','persistence')})
        path = dest/f'predictions_{seed}.npz'
        np.savez_compressed(path,**points)
        bindings.append(path)
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,promotion_allowed=False,
        protocol='Only fit weights changed; fixed source models cloned. Original patient-equal evaluation retained.',
        seeds=results,bindings={str(p):sha256(p) for p in bindings}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        rs = [r['roles'][role] for r in results]
        print(role,{k:float(np.median([r['mae'][k] for r in rs])) for k in points})
        for name in ('own','support','old_full','old_own','persistence'):
            cis = [r['comparisons'][name]['full']['ci95'] for r in rs]
            print('full vs',name,'gain/harm',sum(c[1]<0 for c in cis),sum(c[0]>0 for c in cis))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

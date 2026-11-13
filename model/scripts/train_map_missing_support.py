"""Matched numerical ablation for the MAP native-missing tree experiment."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from scripts.train_map_native_missing import missing_view
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_missing_training_20260912.json')
    output = Path('artifacts/map_missing_support_20260912.json')
    dest = root/'tree_missing_support'
    if output.exists() or dest.exists():
        raise ValueError('fresh outputs required')
    ref = json.loads(parent.read_text())
    for report in (ref,json.loads(Path('artifacts/map_native_anchor_tree_20260912.json').read_text())):
        for p,h in report['bindings'].items():
            assert sha256(Path(p)) == h,p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][valid,0]
    current = labels['current'][valid,0]
    x = np.load(root/'tree_trial/features.npy',mmap_mode='r')
    bindings = [parent,Path(__file__),Path('scripts/train_map_native_missing.py')]
    dest.mkdir()
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        fit = roles['fit']
        subjects = q.loc[fit,'subject_id']
        w = 1/subjects.groupby(subjects).transform('size').to_numpy(float)
        w *= len(w)/w.sum()
        pack = joblib.load(root/f'tree_trial/support_{seed}.joblib')
        n,col,allowed,scale = len(pack['variables']),pack['target_index'],pack['allowed'],pack['target_scale']
        model = clone(pack['model'])
        train = missing_view(x[fit],n,col,allowed,'support')
        model.fit(train,(q.truth.to_numpy()[fit]-current[fit])/scale,sample_weight=w)
        del train
        points = dict(np.load(root/f'tree_missing_trial/predictions_{seed}.npz'))
        pred = np.empty(len(q))
        for start in range(0,len(q),4096):
            end = min(len(q),start+4096)
            pred[start:end] = current[start:end]+scale*model.predict(missing_view(x[start:end],n,col,allowed,'support'))
        points['support'] = pred
        scores = {}
        for role in ('evaluation','hospital'):
            m = roles[role]
            f = q.loc[m]
            p = {k:v[m] for k,v in points.items()}
            for name,mae in entry['roles'][role]['mae'].items():
                np.testing.assert_allclose(patient_equal_mae(f,f.truth,p[name]),mae,rtol=1e-12)
            scores[role] = dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},
                comparisons={name:paired_model_errors(f,p,name,seed=20260912) for name in ('support','own')})
        pack.update(model=model,numeric_missing_representation='nan_with_explicit_mask_age')
        path = dest/f'support_{seed}.joblib'
        joblib.dump(pack,path)
        bindings.append(path)
        path = dest/f'predictions_{seed}.npz'
        np.savez_compressed(path,**points)
        bindings.append(path)
        results.append(dict(seed=seed,roles=scores))
        print(f'Seed {seed} support complete',flush=True)
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,promotion_allowed=False,
        seeds=results,bindings={str(p):sha256(p) for p in bindings}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        rs = [r['roles'][role] for r in results]
        print(role,{k:float(np.median([r['mae'][k] for r in rs])) for k in ('own','full','support')})
        for a,b in [('full','support'),('support','own')]:
            cis = [r['comparisons'][b][a]['ci95'] for r in rs]
            print(a,b,sum(c[1]<0 for c in cis),sum(c[0]>0 for c in cis))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

"""Verify actual fitting-patient support of fitted numerical HGB leaves."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_map_native_anchor_tree import feature_view
from whole_body_jepa.multisource_intervals import sha256


def leaves(nodes, x):
    if nodes['is_categorical'].any():
        raise ValueError('numerical trees only')
    ids = np.zeros(len(x),np.int64)
    for _ in range(len(nodes)):
        active = ~nodes['is_leaf'][ids].astype(bool)
        if not active.any():
            return ids
        rows = np.flatnonzero(active)
        selected = nodes[ids[rows]]
        values = x[rows,selected['feature_idx']]
        left = np.where(np.isnan(values),selected['missing_go_to_left'].astype(bool),values<=selected['num_threshold'])
        ids[rows] = np.where(left,selected['left'],selected['right'])
    raise ValueError('invalid tree traversal')


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private = root/'tree_trial'
    parent = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_native_leaf_patients_20260912.json')
    if output.exists():
        raise ValueError('fresh outputs required')
    ref = json.loads(parent.read_text())
    for p,h in ref['bindings'].items():
        assert sha256(Path(p)) == h,p
    valid = np.load(root/'labels.npz')['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    fit = np.load(root/'partition_7.npz')['fit'][valid]
    patient, ids = pd.factorize(q.loc[fit,'uniquepid'])
    weights = 1/np.bincount(patient)[patient]
    weights /= weights.sum()
    features = np.load(private/'features.npy',mmap_mode='r')
    results = {}
    for mode in ('own','full','support'):
        pack = joblib.load(private/f'{mode}_7.joblib')
        model = pack['model']
        x = feature_view(features[fit],len(pack['variables']),pack['target_index'],pack['allowed'],mode)
        supports, rare_mass = [], []
        predicted = np.full(len(x),float(model._baseline_prediction.item()))
        for trees in model._predictors:
            assert len(trees)==1
            nodes = trees[0].nodes
            node_ids = leaves(nodes,x)
            node_counts = np.bincount(node_ids,minlength=len(nodes))
            leaf = nodes['is_leaf'].astype(bool)
            np.testing.assert_array_equal(node_counts[leaf],nodes['count'][leaf])
            pairs = np.unique(node_ids*len(ids)+patient)
            distinct = np.bincount(pairs//len(ids),minlength=len(nodes))
            supports.extend(distinct[leaf].tolist())
            rare_mass.append(float(weights[distinct[node_ids]<5].sum()))
            predicted += nodes['value'][node_ids]
        np.testing.assert_allclose(predicted,model.predict(x),rtol=1e-12,atol=1e-12)
        support = np.asarray(supports)
        results[mode] = dict(trees=len(model._predictors),leaves=len(support),
            distinct_fit_patient_quantiles=np.quantile(support,[0,.1,.25,.5,.75,.9,1]).tolist(),
            leaf_fraction_below5patients=float((support<5).mean()),
            mean_patient_equal_assignment_fraction_below5=float(np.mean(rare_mass)),
            traversal_prediction_and_counts_verified=True)
        print(mode,results[mode],flush=True)
    output.write_text(json.dumps(dict(complete=True,seed=7,fit_patients=len(ids),results=results,
        training_performed=False,runtime_modified=False,
        caveat='Descriptive support, not effect of enforcing minimum distinct patients; ensemble leaves not independent.',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/train_map_native_anchor_tree.py'),
                                            root/'partition_7.npz',private/'features.npy')}),indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

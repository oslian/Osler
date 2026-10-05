"""Describe fitted MAP split gains and source patient support, not attribution."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def split_usage(model, width):
    count = np.zeros(width,int)
    gain = np.zeros(width,float)
    for stage in model._predictors:
        for tree in stage:
            nodes = tree.nodes
            active = nodes[nodes['is_leaf'] == 0]
            np.add.at(count,active['feature_idx'],1)
            np.add.at(gain,active['feature_idx'],np.maximum(active['gain'],0))
    return count,gain


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_missing_training_20260912.json')
    original = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_feature_usage_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(parent.read_text())
    for r in (ref,json.loads(original.read_text())):
        for p,h in r['bindings'].items():
            assert sha256(Path(p)) == h,p
    valid = np.load(root/'labels.npz')['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    x = np.load(root/'tree_trial/features.npy',mmap_mode='r')
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        pack = joblib.load(root/f'tree_missing_trial/full_{seed}.joblib')
        variables = pack['variables']
        n = len(variables)
        lag = x.shape[1]//(3*n)
        roles = dict(np.load(root/f'partition_{seed}.npz'))
        fit = roles['fit'][valid]
        masks = x[fit,lag*n:2*lag*n].reshape(fit.sum(),lag,n).any(axis=1)
        support = pd.DataFrame(masks).groupby(q.loc[fit,'uniquepid'].to_numpy()).any().sum().to_numpy()
        count,gain = split_usage(pack['model'],x.shape[1])
        count = count.reshape(3,lag,n).sum(axis=1)
        gain = gain.reshape(3,lag,n).sum(axis=1)
        normalized = gain/gain.sum()
        results.append(dict(seed=seed,fit_patients=int(q.loc[fit,'uniquepid'].nunique()),
            rare_threshold_is_descriptive=30,
            rare_gain_fraction=float(normalized[:,support<30].sum()),
            blocks={name:float(normalized[i].sum()) for i,name in enumerate(('numbers','masks','ages'))},
            variables=[dict(variable=v,fit_patients=int(support[j]),
                split_count=int(count[:,j].sum()),gain_fraction=float(normalized[:,j].sum()),
                numeric_gain_fraction=float(normalized[0,j]),mask_gain_fraction=float(normalized[1,j]),
                age_gain_fraction=float(normalized[2,j])) for j,v in enumerate(variables)]))
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        scope='In-sample split gain is usage, not causal importance or held-out predictive value.',
        seeds=results,bindings={str(p):sha256(p) for p in (parent,original,Path(__file__))}),indent=2)+'\n')
    print('Rare gain fraction range',min(r['rare_gain_fraction'] for r in results),max(r['rare_gain_fraction'] for r in results))
    print('Median blocks',{b:float(np.median([r['blocks'][b] for r in results])) for b in results[0]['blocks']})
    ranked = [(v,float(np.median([r['variables'][j]['gain_fraction'] for r in results]))) for j,v in enumerate(variables)]
    print('Top variables',sorted(ranked,key=lambda a:-a[1])[:12])


if __name__ == '__main__':
    main()

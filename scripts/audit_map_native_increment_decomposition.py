"""Paired decomposition of numerical and observation-support fitting gains."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.change_evaluation import estimate
from whole_body_jepa.multisource_intervals import sha256


def components(truth, own, support, full):
    eo, es, ef = [np.abs(np.asarray(v)-truth) for v in (own,support,full)]
    return dict(total=ef-eo, support=es-eo, numerical=ef-es,
                numerical_minus_support=ef-2*es+eo)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_tree_fit_gap_20260912.json')
    original = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_native_increment_decomposition_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(parent.read_text())
    for report in (ref,json.loads(original.read_text())):
        for p,h in report['bindings'].items():
            assert sha256(Path(p)) == h, p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    truth = labels['future'][valid,0]
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        pred = dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        parts = components(truth,pred['own'],pred['support'],pred['full'])
        np.testing.assert_allclose(parts['total'],parts['support']+parts['numerical'],atol=1e-12)
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        scores = {}
        for role,m in roles.items():
            scores[role] = {k:estimate(q.loc[m],v[m],seed=20260912,repetitions=20000)
                            for k,v in parts.items()}
            assert scores[role]['total'] == entry['roles'][role]['versus_own']['full']
            assert scores[role]['numerical'] == entry['roles'][role]['versus_support']['full']
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True,training_performed=False,runtime_modified=False,
        scope='Ordered error contrasts, not additive biological causation or fresh confirmation',
        seeds=results,bindings={str(p):sha256(p) for p in (parent,original,Path(__file__))}),indent=2)+'\n')
    for role in roles:
        rs = [r['roles'][role] for r in results]
        print(role,{k:dict(median=float(np.median([r[k]['mean'] for r in rs])),
                          negative=sum(r[k]['ci95'][1]<0 for r in rs),
                          positive=sum(r[k]['ci95'][0]>0 for r in rs)) for k in parts})


if __name__ == '__main__':
    main()

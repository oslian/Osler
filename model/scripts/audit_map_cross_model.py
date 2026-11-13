"""Matched outcome comparison; not an isolated architecture ablation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parents=[Path('artifacts/native_bedside_map_training_20260911.json'),Path('artifacts/map_summary_components_20260912.json')]
    out=Path('artifacts/map_cross_model_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    for parent in parents:
        for p,h in json.loads(parent.read_text())['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError(f'changed dependency {p}')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    labels=dict(np.load(root/'bedside_future_labels.npz'));col=labels['variables'].tolist().index('map')
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];items=[]
    for seed in (7,19,31,43,59,71,83):
        neural=np.load(root/f'bedside_map_predictions_{seed}.npz')
        tree=np.load(root/f'map_summary_components_{seed}.npz')
        np.testing.assert_array_equal(neural['persistence'],tree['persistence'])
        points={'jepa_own':neural['own'],'jepa_full':neural['bedside'],
                'tree_own_summary':tree['summary_own'],'tree_full_summary':tree['summary_full']}
        roles=dict(np.load(root/f'partition_{seed}.npz'));item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role][valid]&labels['paired'][:,col]
            item['roles'][role]={ref:paired_model_errors(q.loc[m],{k:v[m] for k,v in points.items()},ref,
                seed=20260912,repetitions=20000) for ref in ('jepa_own','jepa_full')}
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,seeds=items,
        bindings={str(p):sha256(p) for p in parents+[Path(__file__),Path('whole_body_jepa/change_evaluation.py')]}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for ref in ('jepa_own','jepa_full'):
            for model in ('tree_own_summary','tree_full_summary'):
                s=[i['roles'][role][ref][model] for i in items]
                print(role,model,ref,'median delta',np.median([a['mean'] for a in s]),
                      'gain/harm',sum(a['ci95'][1]<0 for a in s),sum(a['ci95'][0]>0 for a in s))


if __name__=='__main__':main()

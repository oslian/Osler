"""Pre-anchor MAP freshness strata; no threshold selection or retraining."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_bedside_map_training_20260911.json')
    out=Path('artifacts/map_freshness_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed dependency')
    labels=dict(np.load(root/'bedside_future_labels.npz'));col=labels['variables'].tolist().index('map')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];eligible=labels['paired'][:,col];age=labels['measurement_age'][:,col]
    groups={'age_at_most_1h':eligible&(age<=1.),'age_above_1h':eligible&(age>1.)}
    items=[]
    for seed in (7,19,31,43,59,71,83):
        points=dict(np.load(root/f'bedside_map_predictions_{seed}.npz'))
        roles=dict(np.load(root/f'partition_{seed}.npz'));item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for group,g in groups.items():
                m=roles[role][valid]&g
                item['roles'][role][group]=paired_model_errors(q.loc[m],{k:v[m] for k,v in points.items()},
                    'own',seed=20260912,repetitions=20000)
        items.append(item)
    dt=labels['actual_delta_t'][eligible,col];a=age[eligible]
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        boundary_hours=1.,threshold_selected_by_outcomes=False,
        quantiles=[0,.25,.5,.75,.95,1],age_quantiles=np.quantile(a,[0,.25,.5,.75,.95,1]).tolist(),
        actual_horizon_quantiles=np.quantile(dt,[0,.25,.5,.75,.95,1]).tolist(),
        measurement_gap_quantiles=np.quantile(dt+a,[0,.25,.5,.75,.95,1]).tolist(),
        rows={k:int(v.sum()) for k,v in groups.items()},seeds=items,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),root/'queries.parquet',root/'labels.npz',
            root/'bedside_future_labels.npz',Path('whole_body_jepa/change_evaluation.py'))}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in groups:
            s=[x['roles'][role][group]['bedside'] for x in items]
            print(role,group,'patients',[a['patients'] for a in s],'gain/harm',
                  sum(a['ci95'] is not None and a['ci95'][1]<0 for a in s),
                  sum(a['ci95'] is not None and a['ci95'][0]>0 for a in s))


if __name__=='__main__':main()

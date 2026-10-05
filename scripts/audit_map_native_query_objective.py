"""Check the selected native-query models under their exact value objective."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.audit_map_value_objective import smooth_l1
from scripts.audit_map_expanded_exposure import replay
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    output=Path('artifacts/map_native_query_objective_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    assets=load_release(Path('models'))
    scale=float(assets.scaler.scales[assets.checkpoint['variables'].index('map')])
    q=pd.read_parquet(root/'queries.parquet')
    labels=dict(np.load(root/'labels.npz'))
    items=[]
    for seed in (7,19,31,43,59,71,83):
        report=json.loads(Path(f'artifacts/map_native_query_training_{seed}_20260912.json').read_text())
        roles=dict(np.load(root/f'partition_{seed}.npz'))
        fit=roles['fit']&labels['paired'][:,0]
        pools=[np.asarray(g.index) for _,g in q.loc[fit].groupby('uniquepid',sort=True)]
        counts=replay(pools,seed)
        seen=[]
        for arm in ('own','full'):
            epoch=report['arms'][arm]['selected_epoch']
            mask=np.zeros(len(q),bool)
            if epoch:mask[list(counts[epoch-1])]=True
            seen.append(mask)
        groups={role:roles[role]&labels['paired'][:,0] for role in ('fit','tune','evaluation','hospital')}
        groups.update(both_seen=fit&seen[0]&seen[1],neither_seen=fit&~seen[0]&~seen[1])
        predictions=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        scores={}
        for name,m in groups.items():
            scores[name]={}
            for arm in ('own','full'):
                losses=smooth_l1((predictions[arm][m]-labels['future'][m,0])/scale)
                if not np.isfinite(losses).all():raise ValueError('nonfinite value loss')
                scores[name][arm]=float(pd.Series(losses,index=q.loc[m,'uniquepid']).groupby(level=0).mean().mean())
            scores[name]['full_minus_own']=scores[name]['full']-scores[name]['own']
        items.append(dict(seed=seed,scores=scores))
    summary={name:dict(median_difference=float(np.median([x['scores'][name]['full_minus_own'] for x in items])),
        full_worse=sum(x['scores'][name]['full_minus_own']>0 for x in items)) for name in groups}
    result=dict(complete=True,scale=scale,items=items,summary=summary,
        runtime_modified=False,training_performed=False,
        scope='patient-equal selected value loss only; excludes JEPA latent loss and optimization path',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_map_value_objective.py'),
            Path('scripts/audit_map_expanded_exposure.py'))})
    output.write_text(json.dumps(result,indent=2)+'\n');print(summary)


if __name__=='__main__':main()

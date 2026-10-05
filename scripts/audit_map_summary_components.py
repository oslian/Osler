"""Hold full lag input fixed and vary only the appended summary block."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from whole_body_jepa.event_window_history import select_event_features
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/map_event_history_20260912.json')
    out=Path('artifacts/map_summary_components_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('parent changed')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('map')
    base=dict(np.load(root/'body_arrays_bedside.npz'));rows=base['query_rows']
    labels=dict(np.load(root/'bedside_future_labels.npz'));j=labels['variables'].tolist().index('map')
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=labels['paired'][:,j]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,j];current=labels['current'][:,j];scale=assets.scaler.scales[col]
    y=(q.truth.to_numpy()-current)/scale
    summaries=np.load(root/'map_event_history/summary_features.npz')['features']
    items=[];bindings={str(parent):sha256(parent)}
    for seed in (7,19,31,43,59,71,83):
        rolepath=root/f'partition_{seed}.npz';bindings[str(rolepath)]=sha256(rolepath)
        roles={k:v[valid]&eligible for k,v in dict(np.load(rolepath)).items()};fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(base,rows,q.subject_id,fit)>0
        view=restrict_input_support(base,allowed,assets.model.variable_module_ids.numpy())
        lag=history_features(view,rows,col,'full').astype(float)
        subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points=dict(np.load(root/'map_event_history'/f'predictions_{seed}.npz'))
        item=dict(seed=seed,selection={},roles={})
        for mode in ('own','support','full'):
            # Constant zeros preserve column order/width; only summary content changes.
            a=allowed.copy()
            if mode=='own':a[np.arange(len(a))!=col]=False
            extra=select_event_features(summaries,a,col,'support' if mode=='support' else 'full')
            x=np.concatenate([lag,extra],axis=1);best=None;board=[]
            for leaves in (7,15,31):
                model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                    max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=10.,max_bins=63,
                    early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=w)
                pred=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],pred)
                board.append(dict(leaves=leaves,mae=mae))
                if best is None or mae<best[0]:best=(mae,model)
            points['summary_'+mode]=current+scale*best[1].predict(x);item['selection'][mode]=board
        np.testing.assert_array_equal(points['summary_full'],points['full'])
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m];p={k:v[m] for k,v in points.items()}
            refs={r:paired_model_errors(f,p,r,seed=20260912,repetitions=20000) for r in ('lag_full','summary_own','summary_support')}
            item['roles'][role]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},
                comparisons={f'{a}_vs_{b}':refs[b][a] for a,b in (
                    ('summary_own','lag_full'),('summary_support','summary_own'),('summary_full','summary_support'))})
        path=root/f'map_summary_components_{seed}.npz';np.savez_compressed(path,**points);bindings[str(path)]=sha256(path)
        items.append(item);print('completed',seed,flush=True)
    for p in (Path(__file__),root/'labels.npz',root/'queries.parquet',root/'body_arrays_bedside.npz',root/'bedside_future_labels.npz'):
        bindings[str(p)]=sha256(p)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,
        full_control_exact=True,bindings=bindings),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{k:float(np.median([s['roles'][role]['mae'][k] for s in items])) for k in ('lag_full','summary_own','summary_support','summary_full')},
              {k:(sum(s['roles'][role]['comparisons'][k]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][k]['ci95'][0]>0 for s in items)) for k in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

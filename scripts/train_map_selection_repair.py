"""Old versus expanded MAP fitting, fixed old tune set and matched scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/map_all_query_inputs_20260912.json')
    out=Path('artifacts/map_selection_training_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('inputs changed')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('map')
    base=dict(np.load(root/'map_all_query_inputs.npz'));rows=base['query_rows']
    labels=np.load(root/'map_all_query_labels.npz');eligible=labels['paired'][:,0]
    old=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,0];current=labels['current'][:,0];scale=assets.scaler.scales[col]
    y=(q.truth.to_numpy()-current)/scale;items=[];paths=[parent,Path(__file__)]
    for seed in (7,19,31,43,59,71,83):
        part=root/f'partition_{seed}.npz';paths.append(part);roles=dict(np.load(part))
        tune=roles['tune']&eligible&old;points={'persistence':current};item=dict(seed=seed,selection={},roles={})
        for population in ('old','expanded'):
            fit=roles['fit']&eligible
            if population=='old':fit &= old
            allowed=source_patient_support(base,rows,q.subject_id,fit)>0
            subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
            for kind in ('own','full'):
                channels=allowed.copy()
                if kind=='own':channels[np.arange(len(channels))!=col]=False
                view=restrict_input_support(base,channels,assets.model.variable_module_ids.numpy())
                x=history_features(view,rows,col,'full').astype(float);best=None;board=[]
                for leaves in (7,15,31):
                    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                        max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=10.,max_bins=63,
                        early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=w)
                    mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],current[tune]+scale*model.predict(x[tune]))
                    board.append(dict(leaves=leaves,mae=mae))
                    if best is None or mae<best[0]:best=(mae,model)
                name=population+'_'+kind;points[name]=current+scale*best[1].predict(x)
                item['selection'][name]=dict(fit_rows=int(fit.sum()),fit_patients=int(subjects.nunique()),board=board)
                del x,view
        previous=np.load(root/f'map_raw_predictions_{seed}.npz')
        for kind,name in (('own','own'),('full','real')):
            np.testing.assert_array_equal(points['old_'+kind][old&eligible],previous[name][eligible[old]])
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for subset,subset_mask in (('old',old),('recovered',~old),('all',np.ones(len(q),bool))):
                m=roles[role]&eligible&subset_mask;f=q.loc[m];p={k:v[m] for k,v in points.items()}
                contrasts=[('expanded_own','old_own'),('expanded_full','old_full'),('expanded_full','expanded_own'),('expanded_full','persistence')]
                refs={r:paired_model_errors(f,p,r,seed=20260912,repetitions=20000) for r in set(b for a,b in contrasts)}
                item['roles'][role][subset]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},
                    comparisons={a+'_vs_'+b:refs[b][a] for a,b in contrasts})
        private=root/f'map_selection_predictions_{seed}.npz';np.savez_compressed(private,**points);paths.append(private)
        items.append(item);print('completed',seed,flush=True)
    paths += [root/'queries.parquet',root/'labels.npz',root/'map_all_query_labels.npz',Path('whole_body_jepa/change_evaluation.py')]
    out.write_text(json.dumps(dict(complete=True,status='diagnostic_only',runtime_modified=False,
        old_controls_exact=True,tune_policy='unchanged old source-tune MAP rows',seeds=items,
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for subset in ('old','recovered','all'):
            s=[i['roles'][role][subset] for i in items]
            print(role,subset,{k:float(np.median([a['mae'][k] for a in s])) for k in points},
                {k:(sum(a['comparisons'][k]['ci95'][1]<0 for a in s),sum(a['comparisons'][k]['ci95'][0]>0 for a in s)) for k in s[0]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

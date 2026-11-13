"""Same Mg roles: fixed-lag plus legal event summaries information diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from whole_body_jepa.event_window_history import event_windows, select_event_features
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support,restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');parent=Path('artifacts/native_magnesium_matched_20260911.json');ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_event_history_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    base=dict(np.load(root/'body_arrays.npz'));labels=dict(np.load(root/'multitarget_labels.npz'))
    assets=load_release(Path('models'));variables=list(labels['variables']);col=variables.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];current=labels['current'][:,col];scale=assets.scaler.scales[col]
    y=(q.truth.to_numpy()-current)/scale;rows=base['query_rows'];items=[]
    assert variables == list(assets.checkpoint['variables'])
    raw_path=root/'body_labs.parquet'
    print('Building legal event summaries',flush=True)
    events=event_windows(q,pd.read_parquet(raw_path),variables,assets.scaler.medians,assets.scaler.scales)
    prior=json.loads(Path('artifacts/magnesium_tree_history_20260911.json').read_text())
    private=root/'magnesium_event_history'
    private.mkdir(exist_ok=True)
    for e in ref['seeds']:
        seed=e['seed'];roles={a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()};fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(base,rows,q.subject_id,fit)>0
        view=restrict_input_support(base,allowed,assets.model.variable_module_ids.numpy())
        subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={'persistence':current};item=dict(seed=seed,selection={},roles={})
        for mode in ('own','full','support','lag_own','lag_full'):
            feature_mode=mode.removeprefix('lag_')
            x=history_features(view,rows,col,feature_mode).astype(float)
            if not mode.startswith('lag_'):
                x=np.concatenate([x,select_event_features(events,allowed,col,feature_mode)],axis=1)
            best=None;board=[]
            for alpha in (7,15,31):
                model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=alpha,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=w)
                p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p)
                board.append(dict(max_leaf_nodes=alpha,mae=mae))
                if best is None or mae<best[0]:best=(mae,alpha,model)
            points[mode]=current+scale*best[2].predict(x);item['selection'][mode]=dict(max_leaf_nodes=best[1],board=board)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'mg-tree-{seed}-{role}-{a}-{b}',2000) for a,b in [('full','own'),('full','support'),('full','persistence'),('own','persistence'),('full','lag_full'),('own','lag_own')]})
        previous=next(s for s in prior['seeds'] if s['seed']==seed)
        for role in ('evaluation','hospital'):
            for mode in ('own','full'):
                np.testing.assert_allclose(item['roles'][role]['mae']['lag_'+mode],
                    previous['roles'][role]['mae'][mode],rtol=0,atol=1e-9)
        np.savez_compressed(private/f'predictions_{seed}.npz',**points)
        items.append(item)
        print(f'Seed {seed} complete',flush=True)
    report=dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,
        binding={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_potassium_raw_history.py'),raw_path,root/'queries.parquet',Path('whole_body_jepa/event_window_history.py'),Path('artifacts/magnesium_tree_history_20260911.json'))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()



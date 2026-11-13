"""Same Mg roles: future-event equal training weights diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support,restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');parent=Path('artifacts/native_magnesium_matched_20260911.json');ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_event_weight_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    base=dict(np.load(root/'body_arrays.npz'));labels=dict(np.load(root/'multitarget_labels.npz'))
    assets=load_release(Path('models'));variables=list(labels['variables']);col=variables.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];current=labels['current'][:,col];scale=assets.scaler.scales[col]
    y=(q.truth.to_numpy()-current)/scale;rows=base['query_rows'];items=[]
    for e in ref['seeds']:
        seed=e['seed'];roles={a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()};fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(base,rows,q.subject_id,fit)>0
        view=restrict_input_support(base,allowed,assets.model.variable_module_ids.numpy())
        subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        event_frame=q.loc[fit,['subject_id','stay_id']].copy()
        event_frame['future_id']=labels['future_source_id'][fit,col]
        keys=['subject_id','stay_id','future_id']
        multiplicity=event_frame.groupby(keys)['future_id'].transform('size').to_numpy(float)
        counts=event_frame.drop_duplicates(keys).groupby('subject_id').size()
        ew=1/(multiplicity*event_frame.subject_id.map(counts).to_numpy(float))
        ew*=len(ew)/ew.sum()
        points={'persistence':current};item=dict(seed=seed,selection={},roles={},
            training_support=dict(rows=int(fit.sum()),unique_events=len(event_frame.drop_duplicates(keys)),
                max_event_multiplicity=int(multiplicity.max())))
        for mode in ('own','full','support','event_own','event_full'):
            x=history_features(view,rows,col,mode.removeprefix('event_')).astype(float);best=None;board=[]
            for alpha in (7,15,31):
                model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=alpha,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=ew if mode.startswith('event_') else w)
                p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p)
                board.append(dict(max_leaf_nodes=alpha,mae=mae))
                if best is None or mae<best[0]:best=(mae,alpha,model)
            points[mode]=current+scale*best[2].predict(x);item['selection'][mode]=dict(max_leaf_nodes=best[1],board=board)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'mg-tree-{seed}-{role}-{a}-{b}',2000) for a,b in [('event_full','event_own'),('event_full','full'),('event_own','own'),('event_full','persistence'),('full','own')]})
        prior=json.loads(Path('artifacts/magnesium_tree_history_20260911.json').read_text())
        old=next(s for s in prior['seeds'] if s['seed']==seed)
        for role in ('evaluation','hospital'):
            for mode in ('own','full','support'):
                np.testing.assert_allclose(item['roles'][role]['mae'][mode],old['roles'][role]['mae'][mode],atol=1e-9,rtol=0)
        items.append(item)
        print(f'Seed {seed} complete',flush=True)
    report=dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,
        binding={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_potassium_raw_history.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()



"""Old versus conflict-clean training, on identical clean evaluation rows."""
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
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/joint_conflict_contract_20260911.json');ref=json.loads(parent.read_text())
    for p,d in ref['binding'].items():assert sha256(Path(p))==d,p
    out=Path('artifacts/glucose_conflict_repair_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    old=dict(np.load(root/'multitarget_labels.npz'));new=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    arrays=[dict(np.load(root/p)) for p in ('body_arrays.npz','body_arrays_conflict_checked.npz')]
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));k=list(old['variables']).index('glucose');scale=assets.scaler.scales[k]
    q['truth']=new['future'][:,k];current=new['current'][:,k]
    items=[]
    for seed in (7,19,31,43,59,71,83):
        roles={a:b[valid] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        tune=roles['tune']&new['paired'][:,k]
        points={'persistence':current};item=dict(seed=seed,selection={},roles={})
        for clean,lab,base in zip((False,True),(old,new),arrays):
            fit=roles['fit']&lab['paired'][:,k];rows=base['query_rows']
            allowed=source_patient_support(base,rows,q.subject_id,fit)>0
            view=restrict_input_support(base,allowed,assets.model.variable_module_ids.numpy())
            y=(lab['future'][:,k]-lab['current'][:,k])/scale
            subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
            for mode in ('own','full'):
                x=history_features(view,rows,k,mode).astype(float);best=None;board=[]
                for leaves in (7,15,31):
                    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=leaves,
                        min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
                    model.fit(x[fit],y[fit],sample_weight=w)
                    p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p)
                    board.append(dict(leaves=leaves,mae=mae))
                    if best is None or mae<best[0]:best=(mae,model)
                name=('clean_' if clean else 'old_')+mode
                points[name]=current+scale*best[1].predict(x)
                item['selection'][name]=dict(board=board,fit_rows=int(fit.sum()),fit_patients=int(subjects.nunique()))
        for role in ('evaluation','hospital'):
            m=roles[role]&new['paired'][:,k];f=q.loc[m]
            item['roles'][role]=dict(rows=int(m.sum()),patients=int(f.subject_id.nunique()),
                mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={a+'_vs_'+b:paired(f,points[a][m],points[b][m],f'conflict-{seed}-{role}-{a}-{b}',2000)
                    for a,b in [('clean_full','old_full'),('clean_own','old_own'),('clean_full','clean_own'),('clean_full','persistence')]})
        items.append(item);print('Seed complete',seed,flush=True)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,
        binding={str(p):sha256(p) for p in [parent,Path(__file__),root/'body_arrays.npz',root/'multitarget_labels.npz']}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points})
        print({a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

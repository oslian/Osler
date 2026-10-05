"""Non-neural bedside information test on matched legal fixed-lag histories."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',choices=('glucose','map'),default='glucose')
    args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/native_map_raw_history_20260912.json' if args.target=='map' else 'artifacts/native_bedside_raw_history_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    base=dict(np.load(root/'body_arrays_bedside.npz'));labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    assets=load_release(Path('models'));variables=list(labels['variables']);col=variables.index(args.target)
    if args.target=='map':
        bedside=dict(np.load(root/'bedside_future_labels.npz'));j=bedside['variables'].tolist().index('map')
        for field in ('current','future','paired'):labels[field][:,col]=bedside[field][:,j]
    added=[variables.index(v) for v in ('heart_rate','map','o2sat','respiratory_rate','temperature')]
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=labels['paired'][:,col]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];current=labels['current'][:,col];scale=assets.scaler.scales[col]
    y=(q.truth.to_numpy()-current)/scale;rows=base['query_rows'];items=[]
    for seed in (7,19,31,43,59,71,83):
        roles={a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()};fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(base,rows,q.subject_id,fit)>0
        subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={'persistence':current};item=dict(seed=seed,selection={},roles={})
        for mode in (('own','real','support') if args.target=='map' else ('lab','real','support')):
            channels=allowed.copy()
            if mode=='lab':channels[added]=False
            if mode=='own':channels[np.arange(len(variables))!=col]=False
            view=restrict_input_support(base,channels,assets.model.variable_module_ids.numpy())
            if mode=='support':view['current'][:,np.flatnonzero(np.arange(len(variables))!=col) if args.target=='map' else added]=0
            x=history_features(view,rows,col,'full').astype(float);best=None;board=[]
            for leaves in (7,15,31):
                model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                    max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=10.,max_bins=63,
                    early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=w)
                p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p)
                board.append(dict(leaves=leaves,mae=mae))
                if best is None or mae<best[0]:best=(mae,leaves,model)
            points[mode]=current+scale*best[2].predict(x);item['selection'][mode]=dict(leaves=best[1],board=board)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            contrasts=([('real','own'),('real','support'),('real','persistence'),('own','persistence')]
                       if args.target=='map' else [('real','lab'),('real','support'),('real','persistence'),('support','lab')])
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired_model_errors(f,{k:p[m] for k,p in points.items()},b,
                    seed=20260912,repetitions=20000)[a] for a,b in contrasts})
        if args.target=='map':np.savez_compressed(root/f'map_raw_predictions_{seed}.npz',**points)
        items.append(item);print('completed',seed,flush=True)
    report=dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,target=args.target,
        scope='same queries, labels and roles; fixed-lag snapshots, not identical native event sequences',
        bindings={str(p):sha256(p) for p in [Path(__file__),root/'body_arrays_bedside.npz',
            root/'multitarget_labels_joint_conflict_checked.npz',root/'queries.parquet',root/'labels.npz',
            Path('scripts/audit_potassium_raw_history.py'),Path('whole_body_jepa/change_evaluation.py')]
            +([root/'bedside_future_labels.npz'] if args.target=='map' else [])
            +[root/f'partition_{s}.npz' for s in (7,19,31,43,59,71,83)]})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

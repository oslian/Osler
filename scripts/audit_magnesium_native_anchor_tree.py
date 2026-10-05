"""Same Mg roles: target-native anchor history diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support,restrict_input_support,build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/magnesium_anchor_support_20260911.json');ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_native_anchor_tree_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    private=root/'magnesium_native_anchor_support'
    labels=dict(np.load(private/'labels.npz'));eligible=labels['paired'][:,0]
    q=pd.read_parquet(private/'queries.parquet').loc[eligible].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));variables=list(assets.checkpoint['variables']);col=variables.index('magnesium')
    raw=pd.read_parquet(root/'body_labs.parquet')
    base,rows=build_arrays(q,raw,assets,{s:pd.Timestamp('2100-01-01') for s in q.stay_id.unique()})
    current=labels['current'][eligible,0];q['truth']=labels['future'][eligible,0];scale=assets.scaler.scales[col]
    reconstructed=base['current'][rows,col]*scale+assets.scaler.medians[col]
    np.testing.assert_allclose(current,reconstructed,rtol=1e-6,atol=1e-5)
    original=pd.read_parquet(root/'queries.parquet')
    y=(q.truth.to_numpy()-current)/scale;items=[]
    for seed in (7,19,31,43,59,71,83):
        partition=dict(np.load(root/f'partition_{seed}.npz'))
        roles={a:q.subject_id.isin(set(original.loc[b,'uniquepid'])).to_numpy() for a,b in partition.items()}
        assert np.all(np.stack(list(roles.values())).sum(axis=0)==1)
        fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(base,rows,q.subject_id,fit)>0
        view=restrict_input_support(base,allowed,assets.model.variable_module_ids.numpy())
        subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={'persistence':current};item=dict(seed=seed,selection={},roles={})
        for mode in ('own','full','support'):
            x=history_features(view,rows,col,mode).astype(float);best=None;board=[]
            for alpha in (7,15,31):
                model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=alpha,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed).fit(x[fit],y[fit],sample_weight=w)
                p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],p)
                board.append(dict(max_leaf_nodes=alpha,mae=mae))
                if best is None or mae<best[0]:best=(mae,alpha,model)
            points[mode]=current+scale*best[2].predict(x);item['selection'][mode]=dict(max_leaf_nodes=best[1],board=board)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'mg-tree-{seed}-{role}-{a}-{b}',2000) for a,b in [('full','own'),('full','support'),('full','persistence'),('own','persistence')]})
        np.savez_compressed(private/f'tree_predictions_{seed}.npz',**points)
        items.append(item)
        print(f'Seed {seed} complete',flush=True)
    report=dict(complete=True,runtime_modified=False,status='diagnostic_only',seeds=items,
        binding={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_potassium_raw_history.py'),root/'body_labs.parquet',Path('whole_body_jepa/availability_inputs.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in points},
            {a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':
    with threadpool_limits(limits=4):main()



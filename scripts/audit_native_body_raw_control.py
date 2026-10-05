"""Training-matched state-input ablation of the native replay pilot."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import configure, teacher_target
from whole_body_joint_jepa import _masked_latent_prediction_loss
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from osler_jepa.organ_schema import canonical_organ_adjacency


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/native_body_masked_raw_control_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz')); roles=dict(np.load(root/'partition_7.npz'))
    transforms=dict(np.load(root/'neural_transforms.npz')); hours=[np.array([np.nan])]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            hours.append(np.load(root/'unclipped_clock_shards'/p.name)['event_hours'])
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q)); hours=np.concatenate(hours)
    assets=load_release(Path('models')); jepa=assets.model; variables=assets.checkpoint['variables']
    n=len(variables);col=variables.index('map');median=assets.scaler.medians[col];scale=assets.scaler.scales[col]
    nested=dict(np.load(root/'nested_partition_7.npz'))
    fit=np.flatnonzero(nested['fit'] & labels['paired'][:,0]);views={}
    history=np.unique(np.concatenate([raw.sequences[r] for r in fit]))
    center=np.zeros(n,np.float32);spread=np.ones(n,np.float32);supported=np.zeros(n,bool)
    for i in range(n):
        values=raw.input[history, i][raw.input[history,n+i]>0]
        supported[i]=len(values)>0
        if i!=col and len(values)>1 and np.std(values.astype(float))>=1e-6:
            center[i]=np.median(values);spread[i]=np.std(values.astype(float))
    transforms['7_numeric_center']=center;transforms['7_numeric_scale']=spread;transforms['7_allowed']=supported
    for arm in ('own','full'):
        allowed=transforms['7_allowed'].copy()
        if arm=='own':allowed[np.arange(n)!=col]=False
        views[arm]=transformed_storage(raw,transforms['7_numeric_center'],transforms['7_numeric_scale'],
            allowed,jepa.variable_module_ids.numpy(),fit,own=arm=='own')[0]
    checkpoint=root/'nested_baseline_own_7.pt'
    parent=Path('artifacts/map_nested_baseline_7_20260913.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed baseline dependency')
    expand_history_projection(jepa,6);jepa.belief_filter=NativeEventFilter(jepa.input_dim,jepa.belief_filter.hidden)
    jepa.belief_input_mode='observed_steps';jepa.value_delta_activation=torch.nn.Identity()
    jepa.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);jepa.eval()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(20260913); selected={};data={}
    for role,count in (('fit',64),('tune',32),('evaluation',64),('hospital',64)):
        groups=[np.asarray(g.index) for _,g in q.loc[roles[role]&labels['paired'][:,0]].groupby('subject_id',sort=True)]
        selected[role]=[int(rng.choice(groups[i])) for i in rng.choice(len(groups),count,replace=False)]
        for row in selected[role]:
            x=views['own'].batch([row])
            with torch.no_grad(): base=jepa(*x)
            seq=views['full'].sequences[row];times=torch.from_numpy(hours[seq])
            tokens=torch.from_numpy(views['full'].input[seq].copy());tokens[:,-7:-1]=0
            tokens[:,3*n]=(times/24).float();tokens[:,-1]=(times/168).float()
            anchor=float((pd.Timestamp(q.iloc[row]['t'])-pd.Timestamp('2100-01-01')).total_seconds()/3600)
            args=(tokens,times,anchor,x[1][:,col:col+1],x[2][:,col:col+1],x[5].reshape(1,1),
                base['value'][:,col:col+1].detach(),base['predicted_target_latent'][:,col:col+1].detach())
            data[row]=(args,x[1],torch.tensor([(labels['future'][row,0]-median)/scale],dtype=torch.float32))
    for role,rows in selected.items():np.testing.assert_array_equal(rows,nested['residual_'+role])
    from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target
    counts={}
    for role,rows in selected.items():
        valid=[]
        for r in rows:
            if not np.isfinite(labels['measurement_age'][r,0]) or labels['measurement_age'][r,0]>1:continue
            try: mask_native_target(data[r][0][0],data[r][0][1],variables,'map')
            except ValueError as e:
                if str(e)=='no independent observed history remains':continue
                raise
            valid.append(r)
        selected[role]=valid;counts[role]=len(valid)
        if len(valid)<2:raise ValueError('insufficient supported role')
    q['truth']=labels['current'][:,0]
    population=float(np.median(q.loc[selected['fit'],'truth']))
    rng=np.random.default_rng(20260913)
    donor={}
    for role,rows in selected.items():
        order=np.array(rows)[rng.permutation(len(rows))]
        donor[role]=dict(zip(order,np.roll(order,1)))
        assert all(k!=v for k,v in donor[role].items())
    def predict(model,role,arm):
        values=[]
        with torch.no_grad():
            for r in selected[role]:
                d=donor[role][r] if arm=='placebo' else r
                args=data[d][0]
                values.append(float(model(args[0],args[1],args[2],'map')[0])*scale+median)
        return np.array(values)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    feature={}
    for role,rows in selected.items():
        x=[]
        for r in rows:
            args=data[r][0];tokens,times=mask_native_target(args[0],args[1],variables,'map')
            t=times.numpy();v=tokens.numpy();values=[];support=[]
            for i in range(n):
                observed=v[:,n+i]>0;z=v[observed,i];ts=t[observed]
                if len(z):
                    centered=ts-ts.mean()
                    slope=float(np.dot(centered,z-z.mean())/np.dot(centered,centered)) if np.dot(centered,centered)>0 else 0.
                    values.extend([float(z[-1]),float(z.mean()),float(z.std()),slope])
                    support.extend([1.,float(np.log1p(len(z))),float(args[2]-ts[-1])])
                else:values.extend([0.]*4);support.extend([0.,0.,0.])
            x.append(values+support)
        feature[role]=np.asarray(x)
    parent_mask=Path('artifacts/native_body_masked_pilot_20260913.json')
    prior=json.loads(parent_mask.read_text())
    assert prior['patients']==counts
    for name,h in prior['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed body-pilot dependency')
    models={};boards={};preds={}
    for arm in ('raw','support','placebo'):
        inputs={role:x.copy() for role,x in feature.items()}
        if arm=='support':
            for x in inputs.values():x[:,:4*n]=0
        if arm=='placebo':
            for role,rows in selected.items():
                index={r:i for i,r in enumerate(rows)}
                inputs[role]=feature[role][[index[donor[role][r]] for r in rows]]
        fit_y=q.loc[selected['fit'],'truth'].to_numpy();tune_y=q.loc[selected['tune'],'truth'].to_numpy()
        best=(float(np.mean(abs(population-tune_y))),None);board=[]
        for alpha in (1.,10.,100.,1000.,10000.):
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(inputs['fit'],fit_y)
            mae=float(np.mean(abs(model.predict(inputs['tune'])-tune_y)))
            board.append(dict(alpha=alpha,tune_mae=mae))
            if mae<best[0]:best=(mae,model)
        boards[arm]=dict(board=board,selected='population' if best[1] is None else str(best[1][-1].alpha),tune_mae=best[0])
        preds[arm]={role:np.full(len(selected[role]),population) if best[1] is None else best[1].predict(inputs[role])
                    for role in ('evaluation','hospital')}
    results={}
    for role in ('evaluation','hospital'):
        frame=q.loc[selected[role]];p={a:preds[a][role] for a in preds};p['population']=np.full(len(frame),population)
        results[role]=dict(mae={k:float(np.mean(abs(v-frame.truth.to_numpy()))) for k,v in p.items()},
            comparisons={ref:paired_model_errors(frame,p,ref,seed=20260913,repetitions=20000) for ref in ('population','support','placebo')})
    paths=[parent_mask,Path(__file__),Path('NATIVE_BODY_RAW_CONTROL_PROTOCOL_20260913.md')]
    output.write_text(json.dumps(dict(complete=True,patients=counts,selection=boards,roles=results,
        runtime_modified=False,status='diagnostic_only',task='masked_raw_history_nowcast',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(dict(selection=boards,roles=results)),flush=True)


if __name__=='__main__':main()


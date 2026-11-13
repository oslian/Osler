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
    output=Path('artifacts/native_body_masked_pilot_20260913.json')
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
    logs={};points={};paths=[parent,checkpoint,root/'nested_partition_7.npz',Path(__file__),
        Path('NATIVE_BODY_MASKED_PROTOCOL_20260913.md'),Path('whole_body_jepa/native_body_reconstruction.py'),
        Path('whole_body_jepa/native_organ_state_residual.py'),Path('whole_body_jepa/native_organ_observations.py'),
        Path('whole_body_jepa/physiology_state.py')]
    for arm in ('real','placebo'):
        torch.manual_seed(7)
        model=NativeBodyReconstruction(NativeOrganStateResidual(deepcopy(jepa),variables,('map',))).eval()
        with torch.no_grad():
            model.readout.weight[col].zero_();model.readout.bias[col]=float((population-median)/scale)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        tune_truth=q.loc[selected['tune'],'truth'].to_numpy()
        best_mae=float(np.mean(abs(predict(model,'tune',arm)-tune_truth)))
        best=deepcopy(model.state_dict());selected_epoch=0;log=[];order=np.random.default_rng(7)
        for epoch in range(8):
            for r in order.permutation(selected['fit']):
                d=donor['fit'][r] if arm=='placebo' else r;args=data[d][0]
                optimizer.zero_grad(set_to_none=True)
                prediction=model(args[0],args[1],args[2],'map')
                y=torch.tensor([(q.loc[r,'truth']-median)/scale],dtype=torch.float32)
                loss=torch.nn.functional.smooth_l1_loss(prediction,y)
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            score=float(np.mean(abs(predict(model,'tune',arm)-tune_truth)))
            log.append(dict(epoch=epoch+1,tune_mae=score))
            if score<best_mae:best_mae,best,selected_epoch=score,deepcopy(model.state_dict()),epoch+1
            print(arm,epoch+1,score,flush=True)
        model.load_state_dict(best);logs[arm]=dict(log=log,selected_epoch=selected_epoch,best_tune_mae=best_mae)
        points[arm]={role:predict(model,role,arm) for role in ('evaluation','hospital')}
        path=root/f'native_body_masked_{arm}_7.pt'
        if path.exists():raise ValueError('fresh checkpoint required')
        torch.save(dict(state_dict=best,status='candidate_only',task='masked_map_nowcast',
            numeric_center=center,numeric_scale=spread,allowed_variables=supported,variables=variables),path)
        paths.append(path)
    results={}
    for role in ('evaluation','hospital'):
        frame=q.loc[selected[role]];p={a:points[a][role] for a in points}
        p['population']=np.full(len(frame),population)
        results[role]=dict(mae={k:float(np.mean(abs(v-frame.truth.to_numpy()))) for k,v in p.items()},
            comparisons={ref:paired_model_errors(frame,p,ref,seed=20260913,repetitions=20000) for ref in ('population','placebo')})
    output.write_text(json.dumps(dict(complete=True,patients=counts,logs=logs,roles=results,
        runtime_modified=False,status='candidate_only',task='fresh_map_masked_nowcast',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()

"""Paired source-selected learning pilot for native anatomical state replay."""
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
    output=Path('artifacts/native_organ_constant_latent_20260913.json')
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
    fit=np.flatnonzero(roles['fit'] & labels['paired'][:,0]);views={}
    for arm in ('own','full'):
        allowed=transforms['7_allowed'].copy()
        if arm=='own':allowed[np.arange(n)!=col]=False
        views[arm]=transformed_storage(raw,transforms['7_numeric_center'],transforms['7_numeric_scale'],
            allowed,jepa.variable_module_ids.numpy(),fit,own=arm=='own')[0]
    checkpoint=root/'native_query_own_7.pt'
    parent=Path('artifacts/map_native_query_training_7_20260912.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed baseline dependency')
    expand_history_projection(jepa,6);jepa.belief_filter=NativeEventFilter(jepa.input_dim,jepa.belief_filter.hidden)
    jepa.belief_input_mode='observed_steps';jepa.value_delta_activation=torch.nn.Identity()
    jepa.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']);jepa.eval()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(20260913); selected={};data={}
    for role,count in (('fit',64),('tune',32)):
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
    q['truth']=labels['future'][:,0]
    def predictions(model,rows):
        with torch.no_grad():return np.array([float(model(*data[r][0])['value'][0,0])*scale+median for r in rows])

    report=Path('artifacts/native_organ_replay_trace_20260913.json')
    for name,h in json.loads(report.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed trace dependency')
    paths=[report,Path(__file__)];results={}
    norm=torch.nn.functional.normalize
    for arm in ('none','sparse'):
        path=root/f'canonical_replay_trace_final_{arm}_7.pt'
        model=NativeOrganStateResidual(deepcopy(jepa),variables,('map',)).eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
        paths.append(path);results[arm]={}
        for role,rows in selected.items():
            metrics=[]; corrections=[]
            with torch.no_grad():
                for r in rows:
                    args,current,y=data[r]
                    old=teacher_target(jepa,current,y,col)
                    new=teacher_target(model.jepa,current,y,col)
                    base=args[-1][:,0]
                    pred=model(*args)['future_latent'][:,0]
                    loss=lambda p,t:float(_masked_latent_prediction_loss(p,t,torch.ones(1,dtype=torch.bool)))
                    corrections.append((pred-base).clone())
                    if role=='fit': metrics.append([loss(base,old),loss(pred,old)])
                    else: metrics.append([loss(base,old),loss(pred,old),loss(base+constant,old)])
            if role=='fit':constant=torch.cat(corrections).mean(0,keepdim=True)
            names=['base_fixed_teacher_loss','pred_fixed_teacher_loss']+(['fit_constant_fixed_teacher_loss'] if role=='tune' else [])
            results[arm][role]={k:float(np.mean(np.array(metrics)[:,i])) for i,k in enumerate(names)}
    output.write_text(json.dumps(dict(complete=True,results=results,training_performed=False,
        runtime_modified=False,scope='source-only teacher drift diagnostic, not causal attribution',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()



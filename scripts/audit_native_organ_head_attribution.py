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
    output=Path('artifacts/native_organ_head_attribution_20260913.json')
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
    trace=json.loads(report.read_text())
    for name,h in trace['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed trace dependency')
    results={};paths=[report,Path(__file__)]
    for arm in ('none','sparse'):
        path=root/f'canonical_replay_trace_final_{arm}_7.pt'
        model=NativeOrganStateResidual(deepcopy(jepa),variables,('map',)).eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'])
        paths.append(path);results[arm]={}
        for role in ('fit','tune'):
            rows=selected[role];captured=[]
            handle=model.value_heads[0].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
            with torch.no_grad():original=predictions(model,rows)
            handle.remove();features=torch.cat(captured,0)
            base=np.array([float(data[r][0][-2][0,0])*scale+median for r in rows])
            truth=labels['future'][rows,0]
            np.testing.assert_allclose(np.mean(abs(original-truth)),trace['logs'][arm]['log'][-1][role+'_mae'],rtol=0,atol=1e-5)
            variants={}
            for mode in ('original','no_state','no_anchor','shuffled_state'):
                f=features.clone()
                if mode=='no_state':f[:,:jepa.token_dim]=0
                if mode=='no_anchor':f[:,-2]=0
                if mode=='shuffled_state':
                    permutation=np.random.default_rng(20260913).permutation(len(rows))
                    f[:,:jepa.token_dim]=features[permutation,:jepa.token_dim]
                with torch.no_grad():point=base+model.value_heads[0](f)[:,0].numpy()*scale
                if mode=='original':np.testing.assert_allclose(point,original,rtol=1e-6,atol=1e-4)
                variants[mode]=dict(mae=float(np.mean(abs(point-truth))),
                    mean_abs_output_change=float(np.mean(abs(point-original))),
                    max_abs_output_change=float(np.max(abs(point-original))))
            results[arm][role]=dict(queries=len(rows),variants=variants,
                median_state_coordinate_std=float(features[:,:jepa.token_dim].std(0,unbiased=False).median()),
                anchor_std=float(features[:,-2].std(unbiased=False)))
        del model
    output.write_text(json.dumps(dict(complete=True,role='source_fit_and_tune_only',results=results,
        exact_final_scores_reproduced=True,training_performed=False,runtime_modified=False,
        limitation='Readout intervention, not retraining or causal organ attribution; off-distribution effects possible.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()


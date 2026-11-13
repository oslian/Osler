"""Train support-matched MAP controls under the frozen native-query protocol."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_map_native_queries import predict
from scripts.train_availability_encoder import configure, teacher_target
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_numeric_control import without_other_numbers
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,31,43,59,71,83])
    args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError(f'changed dependency: {name}')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz'));eligible=labels['paired'][:,0]
    q['truth']=labels['future'][:,0]
    transforms=dict(np.load(root/'neural_transforms.npz'))
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q));assets=load_release(Path('models'))
    variables=assets.checkpoint['variables'];n=len(variables);col=variables.index('map')
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    truth=(q.truth.to_numpy()-median)/scale
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        output=Path(f'artifacts/map_native_numeric_control_{seed}_20260912.json')
        checkpoint=root/f'native_numeric_control_{seed}.pt'
        prediction_path=root/f'native_numeric_control_predictions_{seed}.npz'
        if any(p.exists() for p in (output,checkpoint,prediction_path)):
            raise ValueError('overwrite blocked')
        roles=dict(np.load(root/f'partition_{seed}.npz'))
        fit_rows=np.flatnonzero(roles['fit']&eligible)
        tune_rows=np.flatnonzero(roles['tune']&eligible)
        pools=[np.asarray(g.index) for _,g in q.iloc[fit_rows].groupby('subject_id',sort=True)]
        allowed=transforms[f'{seed}_allowed']
        full,ec,es=transformed_storage(raw,transforms[f'{seed}_numeric_center'],
            transforms[f'{seed}_numeric_scale'],allowed,assets.model.variable_module_ids.numpy(),fit_rows,own=False)
        np.testing.assert_array_equal(ec,transforms[f'{seed}_event_center'])
        np.testing.assert_array_equal(es,transforms[f'{seed}_event_scale'])
        storage=without_other_numbers(full,col)
        np.testing.assert_array_equal(storage.input[:,n:-7],full.input[:,n:-7])
        np.testing.assert_array_equal(storage.input[:,-4:],full.input[:,-4:])
        np.testing.assert_array_equal(storage.input[:,col],full.input[:,col])
        for key in ('mask','ages','presence','horizon'):
            np.testing.assert_array_equal(storage.query[key],full.query[key])
        np.testing.assert_array_equal(storage.query['current'][:,col],full.query['current'][:,col])
        del full
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=deepcopy(assets.model);expand_history_projection(model,6)
        native=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        native.load_state_dict(model.belief_filter.state_dict());model.belief_filter=native
        model.belief_input_mode='observed_steps';model.value_delta_activation=torch.nn.Identity()
        params=configure(model,True);optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        best_mae=patient_equal_mae(q.iloc[tune_rows],q.truth.iloc[tune_rows],
            predict(model,storage,tune_rows,col,median,scale))
        best,selected,log=deepcopy(model.state_dict()),0,[]
        print('initial',seed,best_mae,flush=True)
        for epoch in range(32):
            for _ in range(16):
                chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                x=storage.batch(chosen);result=model(*x);y=torch.tensor(truth[chosen],dtype=torch.float32)
                loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(
                    result['predicted_target_latent'][:,col],teacher_target(model,x[1],y,col),
                    torch.ones(len(y),dtype=torch.bool))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();model.update_target_encoder()
            mae=patient_equal_mae(q.iloc[tune_rows],q.truth.iloc[tune_rows],
                predict(model,storage,tune_rows,col,median,scale))
            if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
            log.append(dict(epoch=epoch+1,tune_mae=mae));print(seed,epoch+1,mae,flush=True)
        model.load_state_dict(best)
        points=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        pred=np.full(len(q),np.nan);rows=np.flatnonzero(eligible)
        pred[rows]=predict(model,storage,rows,col,median,scale);points['control']=pred
        evaluation={}
        for role in ('evaluation','hospital'):
            keep=roles[role]&eligible;frame=q.loc[keep];p={k:v[keep] for k,v in points.items()}
            comparisons={f'control_vs_{ref}':paired_model_errors(frame,p,ref,
                seed=20260912,repetitions=20000)['control'] for ref in ('own','full','persistence')}
            evaluation[role]=dict(mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                comparisons=comparisons)
        torch.save(dict(state_dict=best,variables=variables,history_contract='support_numeric_control',
            selected_epoch=selected,allowed_variables=allowed.tolist(),
            numeric_center=transforms[f'{seed}_numeric_center'].tolist(),
            numeric_scale=transforms[f'{seed}_numeric_scale'].tolist(),
            event_center=ec.tolist(),event_scale=es.tolist(),status='candidate_only'),checkpoint)
        np.savez_compressed(prediction_path,control=pred)
        paths=[parent,Path(__file__),Path('whole_body_jepa/native_numeric_control.py'),
            Path('MAP_NATIVE_NUMERIC_CONTROL_PROTOCOL_20260912.md'),checkpoint,prediction_path]
        output.write_text(json.dumps(dict(complete=True,seed=seed,selected_epoch=selected,
            log=log,best_tune_mae=best_mae,roles=evaluation,input_equivalence_passed=True,
            runtime_modified=False,status='candidate_only',bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
        print('complete',seed,json.dumps(evaluation),flush=True)
        del model,storage,optimizer,best,native


if __name__=='__main__':main()

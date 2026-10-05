"""Matched source-selected checkpoint refinement with fresh optimizer state."""
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
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,31,43,59,71,83]);args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz'));eligible=labels['paired'][:,0];q['truth']=labels['future'][:,0]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q));assets=load_release(Path('models'))
    col=assets.checkpoint['variables'].index('map');median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    truth=(q.truth.to_numpy()-median)/scale
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        output=Path(f'artifacts/map_native_query_refine_{seed}_20260912.json')
        if output.exists():raise ValueError('fresh result required')
        roles=dict(np.load(root/f'partition_{seed}.npz'));fit=roles['fit']&eligible
        tune=np.flatnonzero(roles['tune']&eligible)
        pools=[np.asarray(g.index) for _,g in q.loc[fit].groupby('subject_id',sort=True)]
        old=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        points={'old_own':old['own'],'old_full':old['full'],'persistence':old['persistence']}
        arms={};paths=[parent,Path(__file__),Path('scripts/train_map_native_queries.py'),root/f'partition_{seed}.npz']
        for arm in ('own','full'):
            ckpath=root/f'native_query_{arm}_{seed}.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            dest=root/f'native_query_refine_{arm}_{seed}.pt'
            if dest.exists():raise ValueError('checkpoint overwrite blocked')
            view,ec,es=transformed_storage(raw,np.array(ck['numeric_center'],np.float32),
                np.array(ck['numeric_scale'],np.float32),np.array(ck['allowed_variables'],bool),
                assets.model.variable_module_ids.numpy(),np.flatnonzero(fit),own=arm=='own')
            np.testing.assert_array_equal(ec,ck['event_center']);np.testing.assert_array_equal(es,ck['event_scale'])
            torch.manual_seed(seed);rng=np.random.default_rng(seed+10000)
            model=deepcopy(assets.model);expand_history_projection(model,6)
            model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            model.belief_input_mode='observed_steps';model.value_delta_activation=torch.nn.Identity()
            model.load_state_dict(ck['state_dict']);params=configure(model,True)
            optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            before=predict(model,view,tune,col,median,scale)
            np.testing.assert_allclose(before,old[arm][tune],rtol=1e-6,atol=1e-4)
            best_mae=patient_equal_mae(q.iloc[tune],q.truth.iloc[tune],before)
            best=deepcopy(model.state_dict());selected=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    x=view.batch(chosen);result=model(*x);y=torch.tensor(truth[chosen],dtype=torch.float32)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(
                        result['predicted_target_latent'][:,col],teacher_target(model,x[1],y,col),torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();model.update_target_encoder()
                mae=patient_equal_mae(q.iloc[tune],q.truth.iloc[tune],predict(model,view,tune,col,median,scale))
                if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae));print(seed,arm,epoch+1,mae,flush=True)
            model.load_state_dict(best);p=np.full(len(q),np.nan)
            p[eligible]=predict(model,view,np.flatnonzero(eligible),col,median,scale);points[arm]=p
            ck.update(state_dict=best,refine_selected_epoch=selected,optimizer_reset=True)
            torch.save(ck,dest);paths.extend([ckpath,dest]);arms[arm]=dict(selected_epoch=selected,log=log)
            del model,view,optimizer,best,ck
        scored={}
        for role in ('evaluation','hospital'):
            m=roles[role]&eligible;f=q.loc[m];p={k:v[m] for k,v in points.items()}
            contrasts=[('full','own'),('full','old_full'),('own','old_own'),('full','persistence')]
            refs={r:paired_model_errors(f,p,r,seed=20260912,repetitions=20000) for r in set(b for a,b in contrasts)}
            scored[role]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},
                comparisons={a+'_vs_'+b:refs[b][a] for a,b in contrasts})
        path=root/f'native_query_refine_predictions_{seed}.npz';np.savez_compressed(path,**points);paths.append(path)
        output.write_text(json.dumps(dict(complete=True,seed=seed,arms=arms,roles=scored,runtime_modified=False,
            status='candidate_only',optimizer_reset=True,bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
        print('complete',seed,json.dumps(scored),flush=True)


if __name__=='__main__':main()

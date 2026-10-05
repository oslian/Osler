"""Expand MAP fitting only; keep original tune and scored query rows."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support, fit_numeric_rescale, apply_numeric_rescale
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[7,19,31,43,59,71,83]);args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/map_expanded_transforms_20260912.json')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('dependency changed')
    assets=load_release(Path('models'));v=assets.checkpoint['variables'];col=v.index('map')
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=np.load(root/'map_all_query_labels.npz');eligible=labels['paired'][:,0]
    old=np.load(root/'labels.npz')['paired'][:,0];q['truth']=labels['future'][:,0]
    truth=(q.truth.to_numpy()-median)/scale
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        output=Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        if output.exists():raise ValueError('fresh seed output required')
        roles=dict(np.load(root/f'partition_{seed}.npz'));fit=roles['fit']&eligible;tune=roles['tune']&eligible&old
        pools=[np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        previous=np.load(root/f'bedside_map_predictions_{seed}.npz')
        points={'persistence':labels['current'][old,0],'old_own':previous['own'],'old_full':previous['bedside']}
        item=dict(seed=seed,fit_rows=int(fit.sum()),fit_patients=len(pools),arms={},roles={})
        paths=[parent,Path(__file__),root/f'partition_{seed}.npz',root/f'bedside_map_predictions_{seed}.npz']
        for arm in ('own','full'):
            checkpoint=root/f'map_expanded_native_{arm}_{seed}.pt'
            if checkpoint.exists():raise ValueError('checkpoint overwrite blocked')
            base=dict(np.load(root/'map_all_query_inputs.npz'));native=dict(np.load(root/'map_all_native_tokens.npz'))
            arrays,rows,values=native_arrays(base,native,len(v),base['presence'].shape[1])
            centers,scales=fit_numeric_rescale(arrays,rows,fit,col)
            nc=np.zeros(len(v),np.float32);ns=np.ones(len(v),np.float32)
            for var in ('heart_rate','map','o2sat','respiratory_rate','temperature'):
                j=v.index(var);nc[j]=centers[j];ns[j]=scales[j]
            arrays=apply_numeric_rescale(arrays,nc,ns)
            allowed=source_patient_support(base,base['query_rows'],q.subject_id,fit)>0
            events,ec,es=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
            if arm=='own':allowed[np.arange(len(v))!=col]=False;events[:]=0
            view=append_events(restrict_input_support(arrays,allowed,assets.model.variable_module_ids.numpy()),events)
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=deepcopy(assets.model)
            expand_history_projection(model,6);nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            params=configure(model,True);optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            best_mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
            best=deepcopy(model.state_dict());selected=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    x=batch(view,rows[chosen]);result=model(*x);y=torch.tensor(truth[chosen],dtype=torch.float32)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(
                        result['predicted_target_latent'][:,col],teacher_target(model,x[1],y,col),torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
                if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best);points['expanded_'+arm]=predict(model,view,rows[old],col,median,scale)
            torch.save(dict(state_dict=best,variables=v,allowed_variables=allowed.tolist(),numeric_center=nc.tolist(),numeric_scale=ns.tolist(),
                event_center=ec.tolist(),event_scale=es.tolist(),event_width=6,history_contract=arm,
                value_delta_activation='identity',belief_input_mode='observed_steps',status='candidate_only'),checkpoint)
            paths.append(checkpoint);item['arms'][arm]=dict(selected_epoch=selected,log=log)
            print('completed',seed,arm,selected,flush=True)
            del arrays,view,base,native,model,best,optimizer,events,nf
        for role in ('evaluation','hospital'):
            m=(roles[role]&eligible)[old];f=q.loc[old].loc[m];p={k:a[m] for k,a in points.items()}
            contrasts=[('expanded_own','old_own'),('expanded_full','old_full'),('expanded_full','expanded_own'),('expanded_full','persistence')]
            refs={r:paired_model_errors(f,p,r,seed=20260912,repetitions=20000) for r in set(b for a,b in contrasts)}
            item['roles'][role]=dict(mae={k:patient_equal_mae(f,f.truth,a) for k,a in p.items()},comparisons={a+'_vs_'+b:refs[b][a] for a,b in contrasts})
        path=root/f'map_expanded_native_predictions_{seed}.npz';np.savez_compressed(path,**points);paths.append(path)
        output.write_text(json.dumps(dict(complete=True,status='candidate_only',runtime_modified=False,**item,
            bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
        print(json.dumps(item['roles']),flush=True)


if __name__=='__main__':main()

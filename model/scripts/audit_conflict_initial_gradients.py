"""Same initial model and source-fit batch: gradient response to input repair."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch, configure, teacher_target
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    output=Path('artifacts/native_conflict_initial_gradients_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    reference=json.loads(parent.read_text())
    for name,h in reference['binding'].items():
        if sha256(Path(name))!=h:raise ValueError('parent changed')
    assets=load_release(Path('models'));vs=assets.checkpoint['variables'];n=len(vs);col=vs.index('glucose')
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    y=(labels['future'][:,col]-assets.scaler.medians[col])/assets.scaler.scales[col]
    data={}
    for name,b,t in [('old','body_arrays.npz','native_tokens.npz'),('clean','body_arrays_conflict_checked.npz','native_tokens_conflict_replay.npz')]:
        base=dict(np.load(root/b));data[name]=native_arrays(base,dict(np.load(root/t)),n,base['presence'].shape[1])
    blocks={'all':('',),'history':('belief_filter.',),'mixers':('variable_mixer.','module_mixer.','state_encoder.'),
            'future':('latent_predictor.',),'private':tuple(f'{p}.{col}.' for p in ('target_future_predictors','target_temporal_adapters','value_heads'))}
    result=dict(complete=True,runtime_modified=False,training_performed=False,probes=[],
        scope='source-fit initial gradient sensitivity, not optimizer trajectory or error attribution',
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in (7,19,31,43,59,71,83):
        fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]&labels['paired'][:,col]
        pools=[np.flatnonzero(fit&q.uniquepid.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'uniquepid']))]
        views={}
        for name,(arrays,rows,values) in data.items():
            ck=torch.load(root/f'conflict_repair_{name}_full_{seed}.pt',map_location='cpu',weights_only=False)
            observed=np.isfinite(values)
            e=np.column_stack([np.where(observed,(values-np.array(ck['event_center']))/np.array(ck['event_scale']),0),observed]).astype(np.float32)
            views[name]=(append_events(restrict_input_support(arrays,np.array(ck['allowed_variables']),assets.model.variable_module_ids.numpy()),e),rows)
        torch.manual_seed(seed);model=deepcopy(assets.model);expand_history_projection(model,6)
        nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';configure(model,True)
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        rng=np.random.default_rng(seed)
        for step in range(3):
            chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
            gradients={};losses={}
            for name,(view,rows) in views.items():
                x=batch(view,rows[chosen]);out=model(*x);truth=torch.tensor(y[chosen],dtype=torch.float32)
                target=teacher_target(model,x[1],truth,col)
                loss=F.smooth_l1_loss(out['value'][:,col],truth)+_masked_latent_prediction_loss(out['predicted_target_latent'][:,col],target,torch.ones(len(chosen),dtype=torch.bool))
                gradients[name]=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
                losses[name]=float(loss.detach())
            stats={}
            for block,prefixes in blocks.items():
                vectors={}
                for mode,grads in gradients.items():
                    vectors[mode]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                        for (name,p),g in zip(named,grads) if name.startswith(prefixes)])
                a,b=vectors['old'],vectors['clean'];norm=float(a.norm())
                stats[block]=dict(old_norm=norm,clean_norm=float(b.norm()),
                    relative_change=float((a-b).norm())/max(norm,1e-12),
                    cosine=float(F.cosine_similarity(a,b,dim=0)))
            result['probes'].append(dict(seed=seed,step=step,loss=losses,blocks=stats))
    output.write_text(json.dumps(result,indent=2)+'\n')
    for block in blocks:
        vals=[p['blocks'][block] for p in result['probes']]
        print(block,{k:float(np.median([v[k] for v in vals])) for k in vals[0]})


if __name__=='__main__':main()

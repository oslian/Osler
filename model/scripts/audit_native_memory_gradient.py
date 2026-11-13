"""Source-fit-only sensitivity of trained native history by recency."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import batch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import append_events,expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/native_memory_gradient_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/rate_anchor_jepa_clock_20260911.json').read_text())
    with np.load(root/'body_arrays.npz') as f:base={k:f[k] for k in f.files}
    with np.load(root/'native_clock_tokens.npz') as f:native={k:f[k] for k in f.files}
    if sha256(root/'native_clock_tokens.npz')!=ref['binding'][str(root/'native_clock_tokens.npz')]:raise ValueError('cache changed')
    assets=load_release(Path('models'));n=len(assets.checkpoint['variables']);k=assets.checkpoint['variables'].index('potassium')
    arrays,rows,values=native_arrays(base,native,n,base['presence'].shape[1])
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    torch.set_num_threads(4);items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];path=root/f'jepa_clock_real_{seed}.pt'
        if sha256(path)!=ref['binding'][str(path)]:raise ValueError('checkpoint changed')
        saved=torch.load(path,weights_only=True);model=deepcopy(assets.model)
        expand_history_projection(model,saved['event_width']);model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.eval()
        view=restrict_input_support(arrays,np.array(saved['allowed_variables']),model.variable_module_ids.numpy())
        numbers=np.where(np.isfinite(values),(values-np.array(saved['event_center']))/np.array(saved['event_scale']),0.)
        view=append_events(view,np.column_stack([numbers,np.isfinite(values)]).astype(np.float32))
        fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]
        rng=np.random.default_rng(seed);subjects=np.array(sorted(q.loc[fit,'uniquepid'].unique()))
        chosen=rng.choice(subjects,min(32,len(subjects)),replace=False)
        selected=np.array([rng.choice(np.flatnonzero(fit&q.uniquepid.eq(s).to_numpy())) for s in chosen])
        inputs=list(batch(view,rows[selected]));inputs[0].requires_grad_(True)
        prediction=model(*inputs)['value'][:,k]
        grad=torch.autograd.grad(prediction.sum(),inputs[0])[0].detach().numpy()
        h=inputs[0].detach().numpy();active=h[:,:,n:2*n].any(axis=2)|h[:,:,-2].astype(bool)
        distances=np.cumsum(active[:,::-1],axis=1)[:,::-1]-1
        record=dict(seed=seed,patients=len(chosen),channels={})
        for label,index,mask in [('potassium',k,h[:,:,n+k]>0),('insulin_rate',model.input_dim-5,h[:,:,-2]>0)]:
            magnitude=np.abs(grad[:,:,index])*mask;total=float(magnitude.sum())
            record['channels'][label]=dict(total_absolute_gradient=total,
                last5_fraction=float(magnitude[(distances<5)&active].sum()/total) if total else None,
                earlier_than10_fraction=float(magnitude[(distances>=10)&active].sum()/total) if total else None,
                observed_positions=int(mask.sum()),older_positions=int((mask&(distances>=10)).sum()))
        items.append(record)
    report=dict(complete=True,source_fit_only=True,training_performed=False,runtime_modified=False,seeds=items,
                limitation='Local derivatives in normalized coordinates, not causal importance or forecast utility.',
                binding={str(p):sha256(p) for p in (Path(__file__),Path('artifacts/rate_anchor_jepa_clock_20260911.json'),root/'native_clock_tokens.npz')})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(items,indent=2))


if __name__=='__main__':main()

"""Fixed linear probes locate known-signal representation; not promotion."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from scripts.train_availability_encoder import batch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import append_events,expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');out=Path('artifacts/synthetic_latent_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/rate_anchor_jepa_synthetic_20260911.json').read_text())
    with np.load(root/'body_arrays.npz') as f:base={k:f[k] for k in f.files}
    with np.load(root/'native_tokens.npz') as f:native={k:f[k] for k in f.files}
    if sha256(root/'native_tokens.npz')!=ref['binding'][str(root/'native_tokens.npz')]:raise ValueError('cache changed')
    assets=load_release(Path('models'));n=len(assets.checkpoint['variables']);k=assets.checkpoint['variables'].index('potassium')
    arrays,rows,values=native_arrays(base,native,n,base['presence'].shape[1]);known=np.isfinite(values[:,2])
    values[known,2]=np.random.default_rng(20260911).choice([-1.,1.],int(known.sum()))
    target=[]
    for row in rows:
        h=arrays['history'][row];seen=h[known[h]];target.append(values[seen[-1],2] if len(seen) else 0.)
    target=np.array(target);valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    torch.set_num_threads(4);items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];path=root/f'jepa_synthetic_real_{seed}.pt'
        if sha256(path)!=ref['binding'][str(path)]:raise ValueError('checkpoint changed')
        saved=torch.load(path,weights_only=True);model=deepcopy(assets.model)
        expand_history_projection(model,saved['event_width']);model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.eval()
        view=restrict_input_support(arrays,np.array(saved['allowed_variables']),model.variable_module_ids.numpy())
        nums=np.where(np.isfinite(values),(values-np.array(saved['event_center']))/np.array(saved['event_scale']),0.)
        view=append_events(view,np.column_stack([nums,np.isfinite(values)]).astype(np.float32))
        states={name:[] for name in ('belief_state','context_latent','predicted_target_latent')}
        with torch.inference_mode():
            for a in range(0,len(rows),256):
                result=model(*batch(view,rows[a:a+256]))
                for name in states:states[name].append((result[name][:,k] if name=='predicted_target_latent' else result[name]).numpy())
        roles={key:v[valid] for key,v in dict(np.load(root/f'partition_{seed}.npz')).items()};fit=roles['fit']
        counts=q.loc[fit].groupby('uniquepid').size();weights=1/q.loc[fit,'uniquepid'].map(counts).to_numpy();weights*=len(weights)/weights.sum()
        item=dict(seed=seed,probes={})
        for name,chunks in states.items():
            x=np.concatenate(chunks).astype(float);center=np.average(x[fit],axis=0,weights=weights)
            scale=np.sqrt(np.average((x[fit]-center)**2,axis=0,weights=weights)).clip(1e-6);x=(x-center)/scale
            probe=Ridge(alpha=1.).fit(x[fit],target[fit],sample_weight=weights);p=probe.predict(x)
            item['probes'][name]={}
            for role,mask in roles.items():
                errors=pd.DataFrame({'id':q.loc[mask,'uniquepid'],'mse':(p[mask]-target[mask])**2})
                item['probes'][name][role]=dict(patient_mse=float(errors.groupby('id').mse.mean().mean()),correlation=float(np.corrcoef(p[mask],target[mask])[0,1]))
        items.append(item);print('complete',seed,flush=True)
    out.write_text(json.dumps(dict(complete=True,neural_training=False,runtime_modified=False,seeds=items,probe_alpha=1.),indent=2)+'\n')


if __name__=='__main__':main()

"""Compare native numerical treatment increment before and after transfer."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import append_events,expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/native_fit_gap_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/rate_anchor_jepa_clock_20260911.json').read_text())
    with np.load(root/'body_arrays.npz') as f:base={k:f[k] for k in f.files}
    with np.load(root/'native_clock_tokens.npz') as f:native={k:f[k] for k in f.files}
    if sha256(root/'native_clock_tokens.npz')!=ref['binding'][str(root/'native_clock_tokens.npz')]:raise ValueError('cache changed')
    assets=load_release(Path('models'));n=len(assets.checkpoint['variables']);k=assets.checkpoint['variables'].index('potassium')
    arrays,rows,values=native_arrays(base,native,n,base['presence'].shape[1])
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    torch.set_num_threads(4);items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];points={}
        for arm in ('clock_real','clock_support'):
            path=root/f'jepa_{arm}_{seed}.pt'
            if sha256(path)!=ref['binding'][str(path)]:raise ValueError('checkpoint changed')
            saved=torch.load(path,weights_only=True);model=deepcopy(assets.model)
            expand_history_projection(model,saved['event_width']);model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            view=restrict_input_support(arrays,np.array(saved['allowed_variables']),model.variable_module_ids.numpy())
            numbers=np.where(np.isfinite(values),(values-np.array(saved['event_center']))/np.array(saved['event_scale']),0.)
            if arm=='clock_support':numbers[:]=0
            view=append_events(view,np.column_stack([numbers,np.isfinite(values)]).astype(np.float32))
            points[arm]=predict(model,view,rows,k,assets.scaler.medians[k],assets.scaler.scales[k])
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('fit','tune','evaluation','hospital'):
            mask=roles[role];f=q.loc[mask];mae={a:patient_equal_mae(f,f.truth,p[mask]) for a,p in points.items()}
            if role in ('evaluation','hospital'):
                for arm in points:np.testing.assert_allclose(mae[arm],entry['roles'][role]['all']['mae'][arm],atol=1e-7,rtol=0)
            item['roles'][role]=dict(mae=mae,comparison=paired(f,points['clock_real'][mask],points['clock_support'][mask],f'nativefit-{seed}-{role}',1000))
        items.append(item);print('completed',seed,flush=True)
    report=dict(complete=True,fitting_performed=False,runtime_modified=False,seeds=items,
                binding={str(p):sha256(p) for p in (Path(__file__),Path('artifacts/rate_anchor_jepa_clock_20260911.json'),root/'native_clock_tokens.npz')})
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

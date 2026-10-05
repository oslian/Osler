"""Isolate changed event scale under fixed clean weights and old inputs."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_conflict_scale_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    parents=[Path('artifacts/native_conflict_cross_20260911.json'),Path('artifacts/native_conflict_repair_training_20260911.json')]
    for parent in parents:
        for p,h in json.loads(parent.read_text())['binding'].items():
            if sha256(Path(p))!=h:raise ValueError('parent changed')
    assets=load_release(Path('models')); n=len(assets.checkpoint['variables']); col=assets.checkpoint['variables'].index('glucose')
    base=dict(np.load(root/'body_arrays.npz'))
    arrays,rows,values=native_arrays(base,dict(np.load(root/'native_tokens.npz')),n,base['presence'].shape[1])
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));q['truth']=labels['future'][:,col]
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        binding={str(p):sha256(p) for p in parents+[Path(__file__)]})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in (7,19,31,43,59,71,83):
        old=torch.load(root/f'conflict_repair_old_full_{seed}.pt',map_location='cpu',weights_only=False)
        ck=torch.load(root/f'conflict_repair_clean_full_{seed}.pt',map_location='cpu',weights_only=False)
        model=deepcopy(assets.model);expand_history_projection(model,6)
        model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.load_state_dict(ck['state_dict'])
        observed=np.isfinite(values)
        events=np.column_stack([np.where(observed,(values-np.array(old['event_center']))/np.array(old['event_scale']),0),observed]).astype(np.float32)
        view=restrict_input_support(arrays,np.array(old['allowed_variables']),assets.model.variable_module_ids.numpy())
        p=predict(model,append_events(view,events),rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
        cross=dict(np.load(root/f'conflict_cross_predictions_{seed}.npz'))
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(scale_only=paired(f,p[m],cross['clean_weights_old_input'][m],f'scale-{seed}-{role}',1000),
                weights_only=paired(f,p[m],cross['old_weights_old_input'][m],f'weights-{seed}-{role}',1000))
        result['seeds'].append(item)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for contrast in ('scale_only','weights_only'):
            cs=[s['roles'][role][contrast] for s in result['seeds']]
            print(role,contrast,'gain',sum(c['ci95'][1]<0 for c in cs),'harm',sum(c['ci95'][0]>0 for c in cs),
                  'median_effect',float(np.median([c['mean'] for c in cs])))


if __name__=='__main__':main()

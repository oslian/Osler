"""Frozen-weight old/new input replay with checkpoint-bound preprocessing."""
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
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    output=Path('artifacts/native_conflict_cross_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    reference=json.loads(parent.read_text())
    for name,digest in reference['binding'].items():
        if sha256(Path(name))!=digest: raise ValueError('parent changed')
    assets=load_release(Path('models')); n=len(assets.checkpoint['variables']); col=assets.checkpoint['variables'].index('glucose')
    data={}
    for mode,b,t in [('old','body_arrays.npz','native_tokens.npz'),('clean','body_arrays_conflict_checked.npz','native_tokens_conflict_replay.npz')]:
        base=dict(np.load(root/b)); data[mode]=native_arrays(base,dict(np.load(root/t)),n,base['presence'].shape[1])
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz')); eligible=labels['paired'][:,col];q['truth']=labels['future'][:,col]
    result=dict(complete=False,runtime_modified=False,training_performed=False,seeds=[],
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in (7,19,31,43,59,71,83):
        checkpoints={m:torch.load(root/f'conflict_repair_{m}_full_{seed}.pt',map_location='cpu',weights_only=False) for m in ('old','clean')}
        entry=dict(seed=seed,preprocessing_equal={k:bool(np.array_equal(checkpoints['old'][k],checkpoints['clean'][k]))
            for k in ('allowed_variables','event_center','event_scale')},roles={})
        points={}
        for mode,ck in checkpoints.items():
            model=deepcopy(assets.model);expand_history_projection(model,6)
            model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            model.load_state_dict(ck['state_dict'])
            for input_mode,(arrays,rows,values) in data.items():
                observed=np.isfinite(values)
                event=np.column_stack([np.where(observed,(values-np.array(ck['event_center']))/np.array(ck['event_scale']),0.),observed]).astype(np.float32)
                view=restrict_input_support(arrays,np.array(ck['allowed_variables']),assets.model.variable_module_ids.numpy())
                view=append_events(view,event)
                key=f'{mode}_weights_{input_mode}_input'
                points[key]=predict(model,view,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
                if mode==input_mode:
                    expected=np.load(root/f'conflict_repair_predictions_{seed}.npz')[f'{mode}_full']
                    np.testing.assert_array_equal(points[key],expected)
        roles={a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            entry['roles'][role]={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'cross-{seed}-{role}-{a}-{b}',1000)
                for a,b in [('old_weights_clean_input','old_weights_old_input'),
                            ('clean_weights_clean_input','clean_weights_old_input'),
                            ('clean_weights_old_input','old_weights_old_input')]}
        path=root/f'conflict_cross_predictions_{seed}.npz';np.savez_compressed(path,**points)
        result['binding'][str(path)]=sha256(path);result['seeds'].append(entry)
        print('complete',seed,entry['preprocessing_equal'],flush=True)
    result['complete']=True;output.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for name in result['seeds'][0]['roles'][role]:
            cs=[s['roles'][role][name] for s in result['seeds']]
            print(role,name,'gain',sum(c['ci95'][1]<0 for c in cs),'harm',sum(c['ci95'][0]>0 for c in cs),
                  'median_effect',float(np.median([c['mean'] for c in cs])))


if __name__=='__main__':main()

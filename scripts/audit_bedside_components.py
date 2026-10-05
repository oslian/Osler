"""Fixed-weight bedside number versus observation-support interventions."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.availability_inputs import restrict_input_support, apply_numeric_rescale
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_bedside_components_20260911.json')
    if output.exists():raise ValueError('fresh outputs required')
    parent=Path('artifacts/native_bedside_rescale_training_20260911.json')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('parent dependency changed')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];n=len(variables);col=variables.index('glucose')
    added=[variables.index(v) for v in ('heart_rate','map','o2sat','respiratory_rate','temperature')]
    base=dict(np.load(root/'body_arrays_bedside.npz'))
    arrays,rows,values=native_arrays(base,dict(np.load(root/'native_tokens_bedside.npz')),n,base['presence'].shape[1])
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));q['truth']=labels['future'][:,col]
    report=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],bindings={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in (7,19,31,43,59,71,83):
        ck=torch.load(root/f'bedside_rescale_bedside_{seed}.pt',map_location='cpu',weights_only=False)
        model=deepcopy(assets.model);expand_history_projection(model,6)
        model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.load_state_dict(ck['state_dict'])
        observed=np.isfinite(values)
        events=np.column_stack([np.where(observed,(values-np.array(ck['event_center']))/np.array(ck['event_scale']),0),observed]).astype(np.float32)
        transformed=apply_numeric_rescale(arrays,np.array(ck['numeric_center'],np.float32),np.array(ck['numeric_scale'],np.float32))
        allowed=np.array(ck['allowed_variables']);points={}
        for mode in ('original','neutral_numbers','remove_observations'):
            channels=allowed.copy()
            if mode=='remove_observations':channels[added]=False
            view=restrict_input_support(transformed,channels,assets.model.variable_module_ids.numpy())
            if mode=='neutral_numbers':
                view['current'][:,added]=0;view['input'][:,added]=0
            points[mode]=predict(model,append_events(view,events),rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
            del view
        expected=np.load(root/f'bedside_rescale_predictions_{seed}.npz')['bedside']
        np.testing.assert_array_equal(points['original'],expected)
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,original_exact=True,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]={mode:paired(f,points[mode][m],points['original'][m],f'bedsidecomponent-{seed}-{role}-{mode}',1000)
                for mode in ('neutral_numbers','remove_observations')}
        report['seeds'].append(item)
        print('completed',seed,flush=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for mode in ('neutral_numbers','remove_observations'):
            cs=[s['roles'][role][mode] for s in report['seeds']]
            print(role,mode,'gain',sum(c['ci95'][1]<0 for c in cs),'harm',sum(c['ci95'][0]>0 for c in cs),
                'median',float(np.median([c['mean'] for c in cs])))


if __name__=='__main__':main()

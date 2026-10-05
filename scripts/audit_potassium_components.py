"""Fixed-weight source evaluation component diagnosis, not deployment routing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.audit_encoder_groups import neutralize_component
from scripts.train_availability_encoder import predict
from scripts.train_potassium_matched_encoder import three_way_partition
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import build_arrays, restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    training=Path('artifacts/potassium_three_way_20260911.json')
    trained=json.loads(training.read_text())
    if not trained.get('complete'):raise ValueError('unfinished training')
    label_path=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    raw_paths=[Path('/private/tmp/availability_jepa_eicu_labs.parquet'),Path('/private/tmp/encoder_source_expansion_20260910/labs.parquet')]
    for p in [label_path]+raw_paths:
        if sha256(p)!=trained['binding'][str(p)]:raise ValueError('source data changed')
    labels=np.load(label_path);assets=load_release(Path('models'))
    variables=assets.checkpoint['variables'];col=variables.index('potassium')
    if list(labels['variables'])!=variables:raise ValueError('schema mismatch')
    frame=pd.DataFrame(dict(stay_id=labels['stay_id'],subject_id=labels['subject_id'],t=labels['query_time'],truth=labels['future'][:,col]))
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    arrays,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    report=dict(status='diagnostic_only',runtime_modified=False,promotion_allowed=False,external_data_read=False,
        binding={str(p):sha256(p) for p in [training,label_path,Path(__file__),Path('POTASSIUM_COMPONENT_PROTOCOL_20260911.md')]},seeds=[])
    folder=Path('/private/tmp/potassium_three_way_20260911')
    ids=assets.model.variable_module_ids.numpy()
    for entry in trained['seeds']:
        seed=entry['seed'];partition=np.load(folder/f'partition_{seed}.npz')
        calculated=three_way_partition(frame.subject_id,labels['paired'][:,col],seed)
        for key,value in zip(('fit','tune','evaluation'),calculated):np.testing.assert_array_equal(value,partition[key])
        selected=partition['evaluation'];f=frame.loc[selected];r=rows[selected]
        path=folder/f'whole_{seed}.pt';report['binding'][str(path)]=sha256(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['variables']!=variables or saved['target']!='potassium':raise ValueError('checkpoint schema')
        model=assets.model;model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity()
        model.belief_input_mode='observed_steps'
        allowed=np.asarray(saved['allowed_variables'],bool)
        a=restrict_input_support(arrays,allowed,ids)
        def run(view):return predict(model,view,r,col,assets.scaler.medians[col],assets.scaler.scales[col])
        full=run(a);np.testing.assert_allclose(full,np.load(folder/f'whole_{seed}_evaluation.npy'),atol=1e-6,rtol=0)
        own=np.load(folder/f'own_{seed}_evaluation.npy')
        mask=allowed.copy();mask[col]=False
        result=dict(seed=seed,patients=int(f.subject_id.nunique()),components={})
        for component in ('value','age','drop'):
            changed=restrict_input_support(a,allowed & ~mask,ids) if component=='drop' else neutralize_component(a,mask,component)
            for key in ('current','mask','ages'):np.testing.assert_array_equal(changed[key][:,col],a[key][:,col])
            p=run(changed)
            result['components'][component]=dict(mae=patient_equal_mae(f,f.truth,p),
                mean_absolute_shift=float(np.abs(p-full).mean()),
                vs_full=paired(f,p,full,f'potassium-components-{seed}-{component}',1000),
                vs_own=paired(f,p,own,f'potassium-components-own-{seed}-{component}',1000))
        report['seeds'].append(result)
    report['complete']=True
    Path('artifacts/potassium_components_20260911.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

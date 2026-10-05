"""Frozen non-target numeric/age sensitivity, not learned ablation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_encoder_groups import neutralize_component
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/rate_jepa_components_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    reference=json.loads(Path('artifacts/rate_anchor_jepa_20260911.json').read_text())
    cached=json.loads(Path('artifacts/rate_jepa_hospitals_20260911.json').read_text())
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0];arrays=dict(np.load(root/'body_arrays.npz'));rows=arrays.pop('query_rows')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium');torch.set_num_threads(4)
    report=dict(complete=False,runtime_modified=False,fitting_performed=False,status='off_manifold_sensitivity_not_causality',seeds=[])
    for entry in reference['seeds']:
        seed=entry['seed'];path=root/f'jepa_whole_{seed}.pt'
        if sha256(path)!=reference['binding'][str(path)]:raise ValueError('weights changed')
        ppath=root/f'jepa_predictions_{seed}.npz'
        if sha256(ppath)!=cached['binding'][str(ppath)]:raise ValueError('predictions changed')
        points=dict(np.load(ppath));saved=torch.load(path,weights_only=False,map_location='cpu')
        model=assets.model;model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.load_state_dict(saved['state_dict'])
        allowed=np.asarray(saved['allowed_variables'],bool);cross=allowed.copy();cross[col]=False
        view=restrict_input_support(arrays,allowed,model.variable_module_ids.numpy())
        reproduced=predict(model,view,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
        np.testing.assert_allclose(reproduced,points['whole'],atol=1e-7,rtol=0)
        for component in ('value','age'):
            changed=neutralize_component(view,cross,component)
            points[component]=predict(model,changed,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            mask=roles[role];f=q.loc[mask]
            item['roles'][role]=dict(mae={k:patient_equal_mae(f,f.truth,p[mask]) for k,p in points.items()},comparisons={k:paired(f,points[k][mask],points['whole'][mask],f'ratecomponent-{seed}-{role}-{k}',1000) for k in ('value','age')})
        report['seeds'].append(item)
    report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

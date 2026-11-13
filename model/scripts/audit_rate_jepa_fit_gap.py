"""Locate the trained numerical increment along fixed patient partitions."""
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
    output=Path('artifacts/rate_jepa_fit_gap_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    reference=json.loads(Path('artifacts/rate_anchor_jepa_support_20260911.json').read_text())
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    arrays=dict(np.load(root/'body_arrays.npz'));rows=arrays.pop('query_rows')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    torch.set_num_threads(4)
    report=dict(complete=False,fitting_performed=False,runtime_modified=False,seeds=[])
    for item in reference['seeds']:
        seed=item['seed'];ppath=root/f'jepa_predictions_{seed}.npz';path=root/f'jepa_support_{seed}.pt'
        for p in (path,ppath):
            if sha256(p)!=reference['binding'][str(p)]:raise ValueError('reference changed')
        points=dict(np.load(ppath));saved=torch.load(path,map_location='cpu',weights_only=False)
        model=assets.model;model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
        channels=np.asarray(saved['allowed_variables'],bool);cross=channels.copy();cross[col]=False
        view=neutralize_component(restrict_input_support(arrays,channels,model.variable_module_ids.numpy()),cross,'value')
        points['support']=predict(model,view,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        result=dict(seed=seed,roles={})
        for role in ('fit','tune','evaluation','hospital'):
            mask=roles[role];f=q.loc[mask]
            maes={k:patient_equal_mae(f,f.truth,v[mask]) for k,v in points.items()}
            if role in ('evaluation','hospital'):
                for k in points:
                    if abs(maes[k]-item['roles'][role]['all']['mae'][k])>1e-7:raise ValueError('score reproduction failed')
            result['roles'][role]=dict(mae=maes,patients=int(f.subject_id.nunique()),
                comparisons={f'{a}_vs_{b}':paired(f,points[a][mask],points[b][mask],f'fitgap-{seed}-{role}-{a}-{b}',1000)
                             for a,b in [('whole','support'),('whole','own'),('support','own')]})
        report['seeds'].append(result)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

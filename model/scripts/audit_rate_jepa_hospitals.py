"""Frozen hospital attribution; no tuning or selective hospital exclusion."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/rate_jepa_hospitals_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    refpath=Path('artifacts/rate_anchor_jepa_20260911.json');ref=json.loads(refpath.read_text())
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    arrays=dict(np.load(root/'body_arrays.npz'));rows=arrays.pop('query_rows')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    torch.set_num_threads(4);report=dict(complete=False,fitting_performed=False,runtime_modified=False,seeds=[],binding={str(refpath):sha256(refpath)})
    for entry in ref['seeds']:
        seed=entry['seed'];roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()};points={}
        for arm in ('own','whole'):
            path=root/f'jepa_{arm}_{seed}.pt'
            if sha256(path)!=ref['binding'][str(path)]:raise ValueError('weights changed')
            checkpoint=torch.load(path,weights_only=False,map_location='cpu')
            model=deepcopy(assets.model);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';model.load_state_dict(checkpoint['state_dict'])
            view=restrict_input_support(arrays,np.asarray(checkpoint['allowed_variables'],bool),model.variable_module_ids.numpy())
            points[arm]=predict(model,view,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
        for role in ('evaluation','hospital'):
            m=roles[role]
            for arm in points:
                if abs(patient_equal_mae(q.loc[m],q.loc[m,'truth'],points[arm][m])-entry['roles'][role]['all']['mae'][arm])>1e-7:raise ValueError('reproduction failed')
        m=roles['hospital'];f=q.loc[m]
        d=pd.DataFrame(dict(subject=f.subject_id.to_numpy(),hospital=f.hospitalid.to_numpy(),delta=np.abs(points['whole'][m]-f.truth.to_numpy())-np.abs(points['own'][m]-f.truth.to_numpy())))
        hospitals={}
        for h in sorted(f.hospitalid.unique()):
            sub=m&q.hospitalid.eq(h).to_numpy();s=q.loc[sub]
            hospitals[str(h)]=dict(patients=int(s.subject_id.nunique()),comparison=paired(s,points['whole'][sub],points['own'][sub],f'ratehospital-{seed}-{h}',1000))
        # Recompute patient averages after removing each hospital; cross-hospital
        # patients are not counted twice in the pooled mean.
        leave={str(h):float(d.loc[d.hospital.ne(h)].groupby('subject').delta.mean().mean()) for h in f.hospitalid.unique()}
        report['seeds'].append(dict(seed=seed,hospitals=hospitals,leave_one_hospital_mean=leave,pooled_patient_mean=float(d.groupby('subject').delta.mean().mean())))
        path=root/f'jepa_predictions_{seed}.npz';np.savez_compressed(path,**points);report['binding'][str(path)]=sha256(path)
    report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

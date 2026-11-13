"""Source-fit local gradient agreement: numerical loss versus latent JEPA loss."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch,configure,teacher_target
from scripts.train_release_masked_abc import CONTEXT_PREFIXES
from scripts.audit_event_aware_patient_state import patient_partition
from whole_body_jepa.availability_inputs import build_arrays,restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import SEEDS,sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def agreement(model,inputs,y,col):
    out=model(*inputs)
    losses=(F.smooth_l1_loss(out['value'][:,col],y),_masked_latent_prediction_loss(
        out['predicted_target_latent'][:,col],teacher_target(model,inputs[1],y,col),torch.ones(len(y),dtype=torch.bool)))
    params=[p for n,p in model.named_parameters() if n.startswith(CONTEXT_PREFIXES) and p.requires_grad]
    grads=[]
    for i,loss in enumerate(losses):
        g=torch.autograd.grad(loss,params,retain_graph=i==0,allow_unused=True)
        grads.append(torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for p,v in zip(params,g)]))
    a,b=grads;denom=a.norm()*b.norm()
    return dict(value_loss=float(losses[0].detach()),jepa_loss=float(losses[1].detach()),
        cosine=float(a.dot(b)/denom) if denom>0 else None,
        jepa_to_value_norm=float(b.norm()/a.norm()) if a.norm()>0 else None,
        combined_value_alignment=float(a.dot(a+b)/(a.norm()* (a+b).norm())) if a.norm()*(a+b).norm()>0 else None)


def main():
    torch.set_num_threads(4)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    path=Path('/private/tmp/auxiliary_supervision_20260910.npz');labels=np.load(path)
    frame=pd.DataFrame(dict(stay_id=labels['stay_id'],subject_id=labels['subject_id'],t=labels['query_time']))
    old=frame.subject_id.isin(pd.read_parquet('/private/tmp/availability_glucose_development.parquet').subject_id).to_numpy()
    raw=pd.concat([pd.read_parquet('/private/tmp/availability_jepa_eicu_labs.parquet'),
        pd.read_parquet('/private/tmp/encoder_source_expansion_20260910/labs.parquet')],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    arrays,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    report=dict(status='source_gradient_diagnostic',runtime_modified=False,external_data_read=False,
        binding={str(path):sha256(path),str(Path(__file__)):sha256(Path(__file__))},results=[])
    for target,folder,pattern in (('potassium','potassium_matched_encoder_20260911','{arm}_{seed}.pt'),
        ('glucose','encoder_auxiliary_budget_weights_20260910','glucose_only_{seed}.pt')):
        col=variables.index(target)
        for seed in SEEDS:
            a,_=patient_partition(frame.loc[old,'subject_id'],seed)
            fit=np.ones(len(frame),bool);fit[old]=a;fit&=labels['paired'][:,col]
            pools=[np.flatnonzero(fit & frame.subject_id.eq(s).to_numpy()) for s in sorted(set(frame.loc[fit,'subject_id']))]
            for arm in (('own','whole') if target=='potassium' else ('whole',)):
                p=Path('/private/tmp')/folder/pattern.format(arm=arm,seed=seed)
                c=torch.load(p,map_location='cpu',weights_only=False);report['binding'][str(p)]=sha256(p)
                model=deepcopy(assets.model);model.load_state_dict(c['state_dict'])
                model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps';configure(model,True)
                view=restrict_input_support(arrays,c['allowed_variables'],assets.model.variable_module_ids.numpy())
                rng=np.random.default_rng(seed+3401);probes=[]
                for _ in range(3):
                    selected=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    y=(labels['future'][selected,col]-assets.scaler.medians[col])/assets.scaler.scales[col]
                    probes.append(agreement(model,batch(view,rows[selected]),torch.tensor(y,dtype=torch.float32),col))
                report['results'].append(dict(target=target,arm=arm,seed=int(seed),probes=probes))
    Path('artifacts/objective_alignment_20260911.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

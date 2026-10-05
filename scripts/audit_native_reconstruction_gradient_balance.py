"""Real-source smoke audit of query-native multi-target reconstruction."""
import json
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');cache=root/'map_native_anchor_support'
    output=Path('artifacts/native_reconstruction_gradient_balance_20260913.json')
    if output.exists():raise ValueError('fresh output required')
    q=pd.read_parquet(cache/'queries.parquet');fit=np.load(cache/'partition_7.npz')['fit']
    patients=sorted(q.loc[fit,'uniquepid'].unique())[:8]
    stays=q.loc[q.uniquepid.isin(patients),'stay_id'].unique()
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    events=pd.concat([original,bedside],ignore_index=True)
    events=events.loc[events.stay_id.isin(stays)&events.source_kind.isin(['lab_available','bedside_available'])]
    anchors=events[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];n=len(variables)
    from whole_body_jepa.history_event_input import expand_history_projection
    from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
    from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
    torch.manual_seed(7);torch.set_num_threads(4)
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    examples={};seen={};rng=np.random.default_rng(20260913)
    stay_patient=q[['stay_id','uniquepid']].drop_duplicates().set_index('stay_id')['uniquepid']
    if not stay_patient.index.is_unique:raise ValueError('stay patient mapping ambiguous')
    counts={v:0 for v in variables};total=0
    for example in reconstruction_examples(events,anchors,variables,assets.scaler.medians,
            assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
        col=example['target_index'];x=example['tokens'];t=example['event_hours']
        if not np.isfinite(example['truth_normalized']) or (t>example['anchor_hour']).any():raise ValueError('illegal example')
        if (x[:,n+col]!=0).any():raise ValueError('unmasked target')
        counts[variables[col]]+=1;total+=1
        patient=stay_patient.loc[anchors.iloc[example['query_position']]['stay_id']]
        key=(patient,col);seen[key]=seen.get(key,0)+1
        if rng.integers(seen[key])==0:examples[key]=example
    parameters=tuple(model.trajectory.layer.body_update_gate.parameters())
    sums={};losses={};supports={}
    for (patient,col),e in examples.items():
        pred=model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[col])
        loss=torch.nn.functional.smooth_l1_loss(pred,torch.tensor([e['truth_normalized']],dtype=pred.dtype))
        grads=torch.autograd.grad(loss,parameters,allow_unused=True)
        flat=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(grads,parameters)]).detach()
        if not torch.isfinite(flat).all():raise ValueError('nonfinite gradient')
        sums[col]=sums.get(col,torch.zeros_like(flat))+flat
        losses[col]=losses.get(col,0.)+float(loss.detach())
        supports[col]=supports.get(col,0)+1
    cols=sorted(sums);g=torch.stack([sums[c]/supports[c] for c in cols])
    norms=g.norm(dim=1);unit=g/norms[:,None].clamp_min(1e-12)
    cos=unit@unit.T;upper=cos[torch.triu(torch.ones_like(cos),diagonal=1).bool()]
    empirical=torch.tensor([counts[variables[c]] for c in cols],dtype=g.dtype);empirical/=empirical.sum()
    checks=[dict(target=variables[c],sampled_patients=supports[c],available_examples=counts[variables[c]],
        mean_loss=losses[c]/supports[c],body_gradient_norm=float(norms[i]),
        event_weighted_gradient_magnitude=float(empirical[i]*norms[i])) for i,c in enumerate(cols)]
    balance=dict(targets=len(cols),sampled_patient_target_pairs=len(examples),
        negative_pairwise_cosine_fraction=float((upper<0).float().mean()),
        median_pairwise_cosine=float(upper.median()),
        event_weighted_joint_norm=float((g*empirical[:,None]).sum(0).norm()),
        target_balanced_joint_norm=float(g.mean(0).norm()))

    paths=[Path(__file__),Path('whole_body_jepa/native_reconstruction_examples.py'),
        Path('whole_body_jepa/native_history_schedule.py'),Path('whole_body_jepa/native_event_tokens.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),root/'native_events.parquet',root/'qualified_bedside_events.parquet',
        cache/'queries.parquet',cache/'partition_7.npz']
    report=dict(complete=True,patients=len(patients),anchors=len(anchors),examples=total,
        target_counts=counts,training_steps=0,checks=checks,balance=balance,weights_saved=False,efficacy_evaluated=False,runtime_modified=False,
        scope='one reservoir-sampled example per patient/target from eight source patients; initialization-only gradients, not causal diagnosis or efficacy',
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()



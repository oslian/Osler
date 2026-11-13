"""Matched patient/target sampling of one shared native body state."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');cache=root/'map_native_anchor_support'
    output=Path('artifacts/native_reconstruction_sampling_20260913.json')
    private=cache/'multitarget_sampling_7'
    if output.exists() or private.exists():raise ValueError('fresh experiment required')
    private.mkdir()
    torch.manual_seed(7);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    q=pd.read_parquet(cache/'queries.parquet');partition=dict(np.load(cache/'partition_7.npz'))
    rng=np.random.default_rng(20260913);members={}
    for role,count in [('fit',32),('tune',16),('evaluation',16),('hospital',16)]:
        people=np.array(sorted(q.loc[partition[role],'uniquepid'].unique()))
        members[role]=set(rng.choice(people,count,replace=False))
    if sum(map(len,members.values()))!=len(set.union(*members.values())):raise ValueError('patient overlap')
    stay_patient=q[['stay_id','uniquepid']].drop_duplicates().set_index('stay_id')['uniquepid']
    if not stay_patient.index.is_unique:raise ValueError('ambiguous encounter')
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    events=pd.concat([original,bedside],ignore_index=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    data={};frequencies={}
    for role,people in members.items():
        stays=stay_patient.loc[stay_patient.isin(people)].index
        e=events.loc[events.stay_id.isin(stays)&events.source_kind.isin(['lab_available','bedside_available'])]
        anchors=e[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
        samples={};seen={}
        for example in reconstruction_examples(e,anchors,variables,assets.scaler.medians,assets.scaler.scales,
                assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
            patient=stay_patient.loc[anchors.iloc[example['query_position']]['stay_id']]
            key=(patient,example['target_index']);seen[key]=seen.get(key,0)+1
            if rng.integers(seen[key])==0:samples[key]=example
        data[role]=samples;frequencies[role]=seen
        print('built',role,len(people),len(samples),flush=True)
    targets=[c for c in range(len(variables)) if sum(k[1]==c for k in data['fit'])>=5 and sum(k[1]==c for k in data['tune'])>=3]
    if not targets:raise ValueError('no eligible targets')
    data={r:{k:e for k,e in d.items() if k[1] in targets} for r,d in data.items()}
    torch.save(dict(data=data,frequencies=frequencies,members=members,targets=targets),private/'samples.pt')
    medians={c:float(np.median([e['truth_normalized'] for k,e in data['fit'].items() if k[1]==c])) for c in targets}
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,m in medians.items():initial.readout.bias[c]=m
    def predict(model,role):
        with torch.no_grad():
            return {k:float(model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[k[1]])[0]) for k,e in data[role].items()}
    def score(p,role):
        return float(np.mean([np.mean([abs(p[k]-e['truth_normalized']) for k,e in data[role].items() if k[1]==c]) for c in targets if any(k[1]==c for k in data[role])]))
    keys=list(data['fit']);bytarget={c:[k for k in keys if k[1]==c] for c in targets}
    prob=np.array([frequencies['fit'][k] for k in keys],float);prob/=prob.sum()
    summaries={};predictions={}
    for arm in ('event_weighted','balanced'):
        model=deepcopy(initial);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        initial_score=score(predict(model,'tune'),'tune');best=initial_score;epoch_best=0
        checkpoint=deepcopy(model.state_dict());log=[];draw=np.random.default_rng(7)
        for epoch in range(1,9):
            losses=[]
            for _ in range(64):
                if arm=='event_weighted':key=keys[int(draw.choice(len(keys),p=prob))]
                else:
                    c=int(draw.choice(targets));key=bytarget[c][int(draw.integers(len(bytarget[c])))]
                e=data['fit'][key];opt.zero_grad(set_to_none=True)
                value=model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[key[1]])
                loss=torch.nn.functional.smooth_l1_loss(value,torch.tensor([e['truth_normalized']],dtype=value.dtype))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            tune=score(predict(model,'tune'),'tune')
            log.append(dict(epoch=epoch,tune_mae=tune,draw_loss=float(np.mean(losses))))
            if tune<best:best=tune;epoch_best=epoch;checkpoint=deepcopy(model.state_dict())
            print(arm,epoch,tune,flush=True)
        model.load_state_dict(checkpoint)
        torch.save(dict(state_dict=checkpoint,variables=variables,targets=targets,selected_epoch=epoch_best),private/f'{arm}.pt')
        summaries[arm]=dict(initial_tune=initial_score,best_tune=best,selected_epoch=epoch_best,log=log)
        predictions[arm]={r:predict(model,r) for r in ('evaluation','hospital')}
    results={}
    for role in ('evaluation','hospital'):
        results[role]={}
        for c in targets:
            kk=[k for k in data[role] if k[1]==c]
            if len(kk)<2:continue
            frame=pd.DataFrame(dict(subject_id=[k[0] for k in kk],truth=[data[role][k]['truth_normalized'] for k in kk]))
            p={a:np.array([predictions[a][role][k] for k in kk]) for a in predictions};p['population']=np.full(len(kk),medians[c])
            results[role][variables[c]]=dict(patients=len(kk),normalized_mae={a:float(np.mean(abs(v-frame.truth.to_numpy()))) for a,v in p.items()},
                comparisons={ref:paired_model_errors(frame,p,ref,seed=20260913,repetitions=20000) for ref in ('population','event_weighted')})
    paths=[Path(__file__),Path('NATIVE_RECONSTRUCTION_SAMPLING_PROTOCOL_20260913.md'),Path('whole_body_jepa/native_reconstruction_examples.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/native_organ_state_residual.py'),
        root/'native_events.parquet',root/'qualified_bedside_events.parquet',cache/'queries.parquet',cache/'partition_7.npz',*private.glob('*.pt')]
    report=dict(complete=True,targets=[variables[c] for c in targets],patients={r:len(p) for r,p in members.items()},
        samples={r:len(d) for r,d in data.items()},selection=summaries,roles=results,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print('complete',len(targets),flush=True)


if __name__=='__main__':main()

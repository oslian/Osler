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
    output=Path('artifacts/native_reconstruction_full_coverage_20260913.json')
    private=cache/'multitarget_full_coverage_7'
    if output.exists() or private.exists():raise ValueError('fresh experiment required')
    private.mkdir()
    torch.manual_seed(7);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    parent=Path('artifacts/native_reconstruction_sampling_20260913.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed parent dependency')
    cached=torch.load(cache/'multitarget_sampling_7'/'samples.pt',weights_only=False)
    targets=cached['targets'];members=cached['members'];frequencies=cached['frequencies']
    data={r:{k:e for k,e in d.items() if k[1] in targets} for r,d in cached['data'].items()}
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
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
    steps_per_epoch=len(targets)*max(map(len,bytarget.values()))
    summaries={};predictions={}
    for arm in ('event_weighted','balanced'):
        model=deepcopy(initial);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        initial_score=score(predict(model,'tune'),'tune');best=initial_score;epoch_best=0
        checkpoint=deepcopy(model.state_dict());log=[];draw=np.random.default_rng(7)
        for epoch in range(1,9):
            losses=[]
            schedule=[]
            if arm=='balanced':
                shuffled={c:[bytarget[c][i] for i in draw.permutation(len(bytarget[c]))] for c in targets}
                for round_index in range(max(map(len,bytarget.values()))):
                    for c in draw.permutation(targets):
                        schedule.append(shuffled[c][round_index%len(shuffled[c])])
                if set(schedule)!=set(keys):raise ValueError('incomplete epoch coverage')
            for step in range(steps_per_epoch):
                if arm=='event_weighted':key=keys[int(draw.choice(len(keys),p=prob))]
                else:key=schedule[step]
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
    paths=[parent,Path(__file__),Path('NATIVE_RECONSTRUCTION_FULL_COVERAGE_PROTOCOL_20260913.md'),Path('whole_body_jepa/native_reconstruction_examples.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/native_organ_state_residual.py'),
        root/'native_events.parquet',root/'qualified_bedside_events.parquet',cache/'queries.parquet',cache/'partition_7.npz',*private.glob('*.pt')]
    report=dict(complete=True,steps_per_epoch=steps_per_epoch,targets=[variables[c] for c in targets],patients={r:len(p) for r,p in members.items()},
        samples={r:len(d) for r,d in data.items()},selection=summaries,roles=results,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print('complete',len(targets),flush=True)


if __name__=='__main__':main()


"""Localize final observation/gap effects without changing trained weights."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def trace_final(model,tokens,event_hours,anchor_hour,target):
    trajectory=model.trajectory;jepa=trajectory.jepa;layer=trajectory.layer
    clean,times=mask_native_target(tokens,event_hours,trajectory.variables,target)
    body,organs,presence=encode_native_organ_observations(jepa,trajectory.variables,clean)
    state=layer.initialize(clean.new_zeros((1,jepa.latent)),clean.new_zeros((1,len(trajectory.organ_names),jepa.token_dim)),clean.new_zeros((1,len(trajectory.organ_names))))
    relative=(times-times[0]).to(clean.dtype)
    col=trajectory.variables.index(target)
    stages={}
    for i in range(len(times)):
        delta=relative[i:i+1]-state.elapsed_hours
        if i==len(times)-1:stages['before_last_advance']=float(model.readout(state.body)[0,col])
        state=layer.advance(state,delta)
        if i==len(times)-1:stages['before_last_update']=float(model.readout(state.body)[0,col])
        state=layer.update(state,body[i:i+1],organs[i:i+1],presence[i:i+1],delta)
    stages['after_last_update']=float(model.readout(state.body)[0,col])
    gap=(torch.as_tensor(anchor_hour,device=times.device,dtype=times.dtype)-times[-1]).to(clean.dtype).reshape(1)
    state=layer.advance(state,gap)
    stages['final']=float(model.readout(state.body)[0,col])
    return stages,dict(events=len(times),last_gap_hours=float(gap[0]))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    cp=root/'bounded_site_training/bounded.pt'
    out=Path('artifacts/native_reconstruction_final_update_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    training=json.loads(tp.read_text())
    for p,h in training['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training dependency changed')
    ck=torch.load(cp,weights_only=False)
    if ck['bindings']!=training['bindings']:raise ValueError('Checkpoint mismatch')
    ip,fp,sp=root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,fp,sp)]
    variables,targets=joint['variables'],joint['targets'];cols=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    results={};rng=np.random.default_rng(20260920)
    with torch.inference_mode():
        for group in ('selected','broad'):
            store=SourceStore([s for s in shards if s['group']==group and s['role']=='encoder_tune'],cols)
            per={}
            for c in cols:
                keys=store.groups[c];chosen=[keys[i] for i in rng.permutation(len(keys))[:8]];rows=[]
                for key in chosen:
                    e=store[key];tokens=input_tokens(e,'real',len(variables))
                    stages,metadata=trace_final(model,tokens,e['event_hours'],e['anchor_hour'],variables[c])
                    actual=float(model(tokens,e['event_hours'],e['anchor_hour'],variables[c])[0])
                    if not np.isclose(actual,stages['final'],rtol=0,atol=1e-7):raise ValueError('Trace does not reproduce model')
                    error={k:abs(v-e['truth_normalized']) for k,v in stages.items()}
                    rows.append((error,metadata))
                stages=list(rows[0][0]);mae={s:float(np.mean([r[0][s] for r in rows])) for s in stages}
                deltas={f'{a}->{b}':float(np.mean([r[0][b]-r[0][a] for r in rows])) for a,b in zip(stages,stages[1:])}
                per[variables[c]]=dict(patients=len(rows),mae=mae,paired_stage_error_changes=deltas,
                    nonzero_final_gap_pairs=sum(r[1]['last_gap_hours']>0 for r in rows),
                    mean_history_events=float(np.mean([r[1]['events'] for r in rows])))
            results[group]=per;print('Final update trace',group,'complete',flush=True)
    out.write_text(json.dumps(dict(groups=results,bindings={str(p):sha256(p) for p in (Path(__file__),tp,cp,ip,fp,sp)},runtime_modified=False,promotion_allowed=False,
        limitation='Up to eight source encoder patients per target/group. Intermediate readouts scored retrospectively against anchor truth; not forecasts at earlier times, causal attribution, independent validation or proof of a beneficial deletion. Same state trajectory and frozen weights.'),indent=2)+'\n')
    for g,per in results.items():
        for stage in next(iter(per.values()))['paired_stage_error_changes']:
            print(g,stage,'targets_higher_error',sum(r['paired_stage_error_changes'][stage]>0 for r in per.values()),'macro_change',np.mean([r['paired_stage_error_changes'][stage] for r in per.values()]))


if __name__=='__main__':main()

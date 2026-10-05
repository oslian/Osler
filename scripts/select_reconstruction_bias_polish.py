"""Source-held-site gate before retaining a frozen bias-only correction."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_confirmed_reconstruction_heads import evidence,confirmed
from scripts.polish_reconstruction_bias import shifted_state
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def keep_correction(groups):
    return set(groups)=={'selected','broad'} and all(confirmed({'base':v['comparison']},v['patients']) for v in groups.values())


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    rp=Path('artifacts/native_reconstruction_bias_polish_20260920.json')
    mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    cp=root/'bounded_site_training/bounded.pt';polished=root/'bias_polish_20260920/candidate.pt'
    out=Path('artifacts/native_reconstruction_bias_polish_selection_20260920.json')
    private=root/'bias_polish_selection_20260920';private.mkdir(exist_ok=True)
    if out.exists():raise ValueError('Completed selection')
    previous=json.loads(rp.read_text())
    for p,h in previous['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(polished)!=previous['candidate_sha256']:raise ValueError('Changed candidate')
    bindings={str(p):sha256(p) for p in (Path(__file__),rp,mp,jp,cp,polished)}
    bp=private/'bindings.json'
    if bp.exists() and json.loads(bp.read_text())!=bindings:raise ValueError('Resume mismatch')
    if not bp.exists():bp.write_text(json.dumps(bindings,indent=2)+'\n')
    manifest,joint,parent,candidate=[torch.load(p,weights_only=False) for p in (mp,jp,cp,polished)]
    variables,targets=joint['variables'],joint['targets'];columns={variables.index(t) for t in targets}
    fit_ids={s['patient'] for s in manifest['shards'] if s['role']=='fit'}
    shards=[s for s in manifest['shards'] if s['role']=='encoder_tune']
    if fit_ids & {s['patient'] for s in shards}:raise ValueError('Patient overlap')
    site_sets={role:{site for (g,r,p,c),site in manifest['sample_sites'].items() if r==role} for role in ('fit','encoder_tune')}
    if site_sets['fit'] & site_sets['encoder_tune']:raise ValueError('Hospital overlap')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(parent['best_state_dict'])
    deltas={t:float(candidate['state_dict']['readout.bias'][variables.index(t)]-parent['best_state_dict']['readout.bias'][variables.index(t)]) for t in targets}
    grouped={g:{t:[] for t in targets} for g in ('selected','broad')}
    with torch.inference_mode():
        for i,s in enumerate(shards):
            p=Path(s['path']);cache=private/f'patient_{i:05d}.pt'
            if sha256(p)!=s['sha256']:raise ValueError('Changed sample')
            if cache.exists():
                saved=torch.load(cache,weights_only=False)
                if saved['shard']!=s:raise ValueError('Cache identity')
                rows=saved['rows']
            else:
                data=torch.load(p,weights_only=False)
                if data['patient']!=s['patient']:raise ValueError('Sample identity')
                rows=[]
                for col,e in data['samples'].items():
                    if col not in columns:continue
                    target=variables[col];pred=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],target)[0]);truth=float(e['truth_normalized'])
                    rows.append(dict(target=target,truth=truth,prediction=pred,site=manifest['sample_sites'][s['group'],'encoder_tune',s['patient'],col]))
                temp=cache.with_suffix('.tmp');torch.save(dict(shard=s,rows=rows),temp);temp.replace(cache)
            for r in rows:grouped[s['group']][r['target']].append(r)
            if (i+1)%500==0:print('Source-selection patients',i+1,'/',len(shards),flush=True)
    decisions={};accepted={}
    for target in targets:
        groups={}
        for g,per in grouped.items():
            rows=per[target];x=np.array([r['prediction'] for r in rows]);y=np.array([r['truth'] for r in rows]);delta=abs(x+deltas[target]-y)-abs(x-y)
            groups[g]=dict(patients=len(rows),comparison=evidence(delta,[r['site'] for r in rows]))
        keep=keep_correction(groups)
        decisions[target]=dict(keep=keep,groups=groups)
        if keep:accepted[target]=deltas[target]
    # No evaluation-derived change to this frozen decision.
    selected=private/'candidate.pt';torch.save(dict(state_dict=shifted_state(parent['best_state_dict'],variables,accepted),accepted=accepted,decisions=decisions,bindings=bindings,promotion_allowed=False),selected)
    out.write_text(json.dumps(dict(decisions=decisions,accepted=list(accepted),fit_tune_patient_overlap=0,fit_tune_site_overlap=0,
        bindings=bindings,candidate_sha256=sha256(selected),runtime_modified=False,promotion_allowed=False,
        limitation='Reused encoder-selection hospitals, both source groups must pass existing incremental39-adjusted gate; not a fresh external evaluation or full promotion. Rejected biases remain exactly original. All33trained targets included.'),indent=2)+'\n')
    print('Accepted source corrections',list(accepted),flush=True)


if __name__=='__main__':main()

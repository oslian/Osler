"""Source-only exact per-target SmoothL1 bias polish of one frozen body state."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_bun_bias_stationarity import optimum
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from scripts.audit_confirmed_reconstruction_heads import evidence,confirmed
from scripts.bounded_site_sampling import bounded_weights
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def shifted_state(state,variables,shifts):
    result=deepcopy(state)
    for target,shift in shifts.items():
        result['readout.bias'][variables.index(target)]+=shift
    return result


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    mr=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    er=Path('artifacts/native_reconstruction_bounded_site_evaluation_20260920.json')
    out=Path('artifacts/native_reconstruction_bias_polish_20260920.json')
    private=root/'bias_polish_20260920';private.mkdir(exist_ok=True)
    if out.exists():raise ValueError('Completed experiment')
    mapping,previous=[json.loads(p.read_text()) for p in (mr,er)]
    if sha256(mp)!=mapping['private_sha256']:raise ValueError('Manifest changed')
    for p,h in previous['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Evaluation dependency changed')
    paths=(Path(__file__),cp,mp,jp,mr,er)
    bindings={str(p):sha256(p) for p in paths}
    bp=private/'bindings.json'
    if bp.exists() and json.loads(bp.read_text())!=bindings:raise ValueError('Resume mismatch')
    if not bp.exists():bp.write_text(json.dumps(bindings,indent=2)+'\n')
    ck,manifest,joint=[torch.load(p,weights_only=False) for p in (cp,mp,jp)]
    variables,targets=joint['variables'],joint['targets'];columns={variables.index(t) for t in targets}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    fit={g:{t:[] for t in targets} for g in ('selected','broad')};identities=set()
    shards=[s for s in manifest['shards'] if s['role']=='fit']
    with torch.inference_mode():
        for i,s in enumerate(shards):
            if s['patient'] in identities:raise ValueError('Duplicate patient')
            identities.add(s['patient']);p=Path(s['path']);save=private/f'fit_{i:05d}.pt'
            if sha256(p)!=s['sha256']:raise ValueError('Changed sample')
            if save.exists():
                saved=torch.load(save,weights_only=False)
                if saved['shard']!=s:raise ValueError('Cached shard mismatch')
                rows=saved['rows']
            else:
                sample=torch.load(p,weights_only=False)
                if sample['patient']!=s['patient']:raise ValueError('Patient mismatch')
                rows=[]
                for col,e in sample['samples'].items():
                    if col not in columns:continue
                    target=variables[col]
                    pred=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],target)[0])
                    rows.append(dict(target=target,truth=float(e['truth_normalized']),prediction=pred,
                        site=manifest['sample_sites'][s['group'],'fit',s['patient'],col]))
                temp=save.with_suffix('.tmp');torch.save(dict(shard=s,rows=rows),temp);temp.replace(save)
            for r in rows:fit[s['group']][r['target']].append(r)
            if (i+1)%500==0:print('Source patients',i+1,'/',len(shards),flush=True)
    shifts={};training={}
    for target in targets:
        residuals=[];weights=[]
        for group in fit:
            rows=fit[group][target]
            if not rows:raise ValueError('Unsupported target')
            residuals.extend(r['prediction']-r['truth'] for r in rows)
            w=bounded_weights([r['site'] for r in rows]);weights.extend(w/w.sum()/len(fit))
        r=np.array(residuals);w=np.array(weights);shift,before,after=optimum(r,w);shifts[target]=shift
        original=float(np.average(smooth_l1_from_error(abs(r)),weights=w))
        polished=float(np.average(smooth_l1_from_error(abs(r+shift)),weights=w))
        if polished>original+1e-10 or abs(after)>1e-8:raise ValueError('Optimization failure')
        training[target]=dict(pairs=len(r),shift=shift,gradient_before=before,gradient_after=after,smooth_l1_before=original,smooth_l1_after=polished)
    state=shifted_state(ck['best_state_dict'],variables,shifts)
    # Freeze all decisions before opening evaluation rows.
    candidate=private/'candidate.pt'
    torch.save(dict(state_dict=state,shifts=shifts,bindings=bindings,training=training,promotion_allowed=False),candidate)
    results={}
    for group in ('selected','broad'):
        ep=root/f'bounded_site_evaluation/{group}.pt'
        if sha256(ep)!=previous['bindings'][str(ep)]:raise ValueError('Evaluation rows changed')
        rows=torch.load(ep,weights_only=False)
        if identities & {r['patient'] for r in rows}:raise ValueError('Fit/evaluation overlap')
        per={}
        for target in targets:
            part=[r for r in rows if r['target']==target]
            y=np.array([r['truth'] for r in part]);pred=np.array([r['site_bounded'] for r in part])
            actual_shift=float(state['readout.bias'][variables.index(target)]-ck['best_state_dict']['readout.bias'][variables.index(target)])
            errors=abs(pred-y);new=abs(pred+actual_shift-y)
            if not np.isclose(errors.mean(),previous['groups'][group][target]['mae']['site_bounded'],atol=1e-10,rtol=0):raise ValueError('Baseline reproduction')
            checks=evidence(new-errors,[r['site'] for r in part])
            per[target]=dict(patients=len(part),mae_before=float(errors.mean()),mae_after=float(new.mean()),comparison=checks,
                incremental_gate=confirmed({'original':checks},len(part)))
        results[group]=per
        print(group,dict(before=float(np.mean([r['mae_before'] for r in per.values()])),after=float(np.mean([r['mae_after'] for r in per.values()])),
            numeric_gains=sum(r['mae_after']<r['mae_before'] for r in per.values()),incremental=[t for t,r in per.items() if r['incremental_gate']]),flush=True)
    out.write_text(json.dumps(dict(training=training,groups=results,bindings=bindings,candidate_sha256=sha256(candidate),runtime_modified=False,promotion_allowed=False,
        limitation='All33 source-fit scalar data-risk optima fixed before cached reused evaluation. Incremental patient/site39-adjusted gate only, not complete population/placebo/interval promotion. Six untrained targets unchanged. Shared state and all non-bias weights unchanged.'),indent=2)+'\n')


if __name__=='__main__':main()

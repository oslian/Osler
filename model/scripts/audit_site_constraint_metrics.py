"""Reconcile empirical site SmoothL1 constraints with source and held MAE."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.refit_site_constrained_readout import losses
from whole_body_jepa.multisource_intervals import sha256


def metric_changes(old_residual, new_residual):
    return dict(mae=float(np.mean(abs(new_residual)-abs(old_residual))),
                smooth_l1=float(np.mean(losses(new_residual)-losses(old_residual))))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_site_constrained_readout_20260920.json')
    sp=Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    out=Path('artifacts/native_reconstruction_site_constraint_metrics_20260920.json')
    if out.exists():raise ValueError('Existing report')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    mp=root/'site_balance_manifest_20260920.pt';cp=root/'bounded_site_training/bounded.pt'
    ck=torch.load(cp,weights_only=False);manifest=torch.load(mp,weights_only=False)
    targets=list(parent['targets']);rows={t:[] for t in targets};digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        p=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt'
        digest.update((p.name+sha256(p)).encode())
        if s['role']!='fit':continue
        saved=torch.load(p,weights_only=False)
        if saved['shard']!=s:raise ValueError('Wrong cache identity')
        for r in saved['rows']:rows[r['target']].append(dict(r,group=s['group']))
    if digest.hexdigest()!=parent['cache_digest']:raise ValueError('Changed feature cache')
    result={}
    for target,rr in rows.items():
        p=root/'site_constrained_readout_20260920'/f'{target}.pt'
        if sha256(p)!=parent['coefficient_hashes'][target]:raise ValueError('Changed coefficients')
        w,b=torch.load(p,weights_only=False)['constrained'];col=ck['variables'].index(target)
        x=np.stack([r['body'] for r in rr]).astype(float);y=np.array([r['truth'] for r in rr])
        old=x@ck['best_state_dict']['readout.weight'][col].numpy().astype(float)+float(ck['best_state_dict']['readout.bias'][col])-y
        new=x@w+b-y;groups=np.array([r['group'] for r in rr]);sites=np.array([str(r['site']) for r in rr])
        per={}
        for group in ('selected','broad'):
            gm=groups==group;metrics=[]
            for site in sorted(set(sites[gm])):
                mask=gm&(sites==site)
                metrics.append(dict(patients=int(mask.sum()),**metric_changes(old[mask],new[mask])))
            if max(v['smooth_l1'] for v in metrics)>1e-7:raise ValueError('Source constraint violated')
            harm=[v for v in metrics if v['mae']>1e-7]
            held=parent['targets'][target][group]
            per[group]=dict(source_patients=int(gm.sum()),source_sites=len(metrics),
                source_patient_change=metric_changes(old[gm],new[gm]),
                source_site_equal_mae_change=float(np.mean([v['mae'] for v in metrics])),
                source_mae_harm_sites=len(harm),harm_site_support=[v['patients'] for v in harm],
                max_site_mae_increase=float(max(v['mae'] for v in metrics)),
                held_patient_mae_change=held['mae']['constrained']-held['mae']['original'],
                held_site_mae_change=held['comparisons']['original']['site_mean'])
        result[target]=per
    pure_transfer=[dict(target=t,group=g) for t,per in result.items() for g,v in per.items()
                   if v['held_patient_mae_change']>0 and v['source_mae_harm_sites']==0]
    summary=dict(targets=len(result),group_targets=len(result)*2,
        group_targets_with_source_mae_harm=sum(v['source_mae_harm_sites']>0 for per in result.values() for v in per.values()),
        total_source_sites=sum(v['source_sites'] for per in result.values() for v in per.values()),
        total_source_mae_harm_sites=sum(v['source_mae_harm_sites'] for per in result.values() for v in per.values()),
        held_harm_without_any_source_mae_harm=pure_transfer)
    out.write_text(json.dumps(dict(targets=result,summary=summary,bindings={str(p):sha256(p) for p in (Path(__file__),pp,sp,mp,cp)},
        runtime_modified=False,promotion_allowed=False,
        limitation='Same exact constrained heads; no fitting or selection. Source MAE numeric tolerance1e-7; descriptive site means, not population guarantees or sole-cause proof. Current masked reconstruction, not forecasts.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(json.dumps({t:result[t] for t in ('glucose','ptt','bun')},indent=2))


if __name__=='__main__':main()

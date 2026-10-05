"""Fit-only spectral localization of a same-capacity readout correction."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.bounded_site_sampling import bounded_weights
from scripts.audit_confirmed_reconstruction_heads import evidence
from whole_body_jepa.multisource_intervals import sha256


def spectrum(x,w,correction):
    x=np.asarray(x,dtype=float);w=np.asarray(w,dtype=float);w=w/w.sum()
    mean=np.average(x,axis=0,weights=w)
    _,singular,vh=np.linalg.svd((x-mean)*np.sqrt(w[:,None]),full_matrices=False)
    bottom=max(1,x.shape[1]//4)
    weak=vh[-bottom:].T@(vh[-bottom:]@correction)
    variance=singular**2
    return mean,weak,dict(singular_values=singular.tolist(),weak_dimensions=bottom,
        weak_variance_fraction=float(variance[-bottom:].sum()/variance.sum()),
        weak_correction_norm_fraction=float(weak@weak/(correction@correction)) if correction@correction>0 else 0.,
        effective_rank=float(variance.sum()**2/(variance@variance)))


def main():
    root=Path('data/research_recovery/broad_source_20260917');private=root/'frozen_readout_refit_20260920'
    rp=Path('artifacts/native_reconstruction_frozen_readout_refit_20260920.json');cp=private/'candidate.pt'
    out=Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    previous=json.loads(rp.read_text())
    for p,h in previous['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(cp)!=previous['candidate_sha256']:raise ValueError('Changed readout')
    parent=torch.load(root/'bounded_site_training/bounded.pt',weights_only=False)
    candidate=torch.load(cp,weights_only=False)['state_dict']
    manifest=torch.load(root/'site_balance_manifest_20260920.pt',weights_only=False)
    joint=torch.load(root/'joint_source_manifest.pt',weights_only=False);variables=joint['variables'];targets=joint['targets']
    grouped={role:{g:{t:[] for t in targets} for g in ('selected','broad')} for role in ('fit','encoder_tune')};digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        if s['role'] not in grouped:continue
        p=private/f'features_{i:05d}.pt';digest.update((p.name+sha256(p)).encode())
        data=torch.load(p,weights_only=False)
        if data['shard']!=s:raise ValueError('Identity mismatch')
        for r in data['rows']:grouped[s['role']][s['group']][r['target']].append(r)
    results={}
    with threadpool_limits(limits=4):
        for target in targets:
            col=variables.index(target);oldw=parent['best_state_dict']['readout.weight'][col].numpy().astype(float);oldb=float(parent['best_state_dict']['readout.bias'][col])
            neww=candidate['readout.weight'][col].numpy().astype(float);newb=float(candidate['readout.bias'][col]);correction=neww-oldw
            parts=[grouped['fit'][g][target] for g in ('selected','broad')];x=np.stack([r['body'] for part in parts for r in part]);weights=[]
            for part in parts:
                w=bounded_weights([r['site'] for r in part]);weights.extend(w/w.sum()/2)
            mean,weak,stats=spectrum(x,weights,correction);groups={}
            for g in ('selected','broad'):
                rows=grouped['encoder_tune'][g][target];z=np.stack([r['body'] for r in rows]);y=np.array([r['truth'] for r in rows]);original=np.array([r['prediction'] for r in rows])
                full=z@neww+newb;part=(z-mean)@weak;ablated=full-part
                if not np.isclose(abs(full-y).mean(),previous['source_selection'][target][g]['after'],atol=1e-5,rtol=0):raise ValueError('Refit score mismatch')
                sites=[r['site'] for r in rows]
                groups[g]=dict(patients=len(rows),mae_original=float(abs(original-y).mean()),mae_refit=float(abs(full-y).mean()),mae_weak_correction_removed=float(abs(ablated-y).mean()),
                    ablation_vs_refit=evidence(abs(ablated-y)-abs(full-y),sites),ablation_vs_original=evidence(abs(ablated-y)-abs(original-y),sites),
                    weak_correction_rms=float(np.sqrt(np.mean(part**2))))
            results[target]=dict(spectrum=stats,groups=groups)
    summary={g:dict(original=float(np.mean([r['groups'][g]['mae_original'] for r in results.values()])),refit=float(np.mean([r['groups'][g]['mae_refit'] for r in results.values()])),
        ablated=float(np.mean([r['groups'][g]['mae_weak_correction_removed'] for r in results.values()])),
        ablation_numeric_gains=sum(r['groups'][g]['mae_weak_correction_removed']<r['groups'][g]['mae_refit'] for r in results.values())) for g in ('selected','broad')}
    output=dict(targets=results,summary=summary,cache_digest=digest.hexdigest(),
        bindings={str(p):sha256(p) for p in (Path(__file__),rp,cp)},runtime_modified=False,promotion_allowed=False,
        limitation='Fixed bottom16 of64 fit-weighted SVD directions, not tuned rank. Remove only centered refit correction there, not original weights. Low variance is not proof of noise. Reused source selection; no deployment or new validation.')
    out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(summary,indent=2));print('median weak variance/norm fractions',*[float(np.median([r['spectrum'][k] for r in results.values()])) for k in ('weak_variance_fraction','weak_correction_norm_fraction')])


if __name__=='__main__':main()

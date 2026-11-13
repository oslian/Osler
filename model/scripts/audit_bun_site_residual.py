"""Attach existing exact sample-site provenance to signed BUN residuals."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_bun_conditional_residual import median_interval
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    mp=root/'site_balance_manifest_20260920.pt'
    mr=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    rp=Path('artifacts/native_reconstruction_bun_offset_transfer_20260920.json')
    fp=root/'bun_dynamics_fit_20260920.pt';vp=root/'bun_offset_transfer_20260920.pt'
    jp=root/'joint_source_manifest.pt'
    out=Path('artifacts/native_reconstruction_bun_site_residual_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    mapping,prior=[json.loads(p.read_text()) for p in (mr,rp)]
    if sha256(mp)!=mapping['private_sha256'] or sha256(vp)!=prior['private_sha256']:raise ValueError('Changed cache')
    for p,h in prior['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    manifest,joint,fit,tune=[torch.load(p,weights_only=False) for p in (mp,jp,fp,vp)]
    col=joint['variables'].index('bun');result={}
    for role,data in (('fit',fit),('encoder_tune',tune)):
        result[role]={}
        for group,rows in data.items():
            by_site={}
            for r in rows:
                site=manifest['sample_sites'][(group,role,r['patient'],col)]
                by_site.setdefault(str(site),[]).append(r['truth']-r['predictions']['full'])
            counts=np.array([len(v) for v in by_site.values()],dtype=float)
            result[role][group]=dict(patients=len(rows),sites=len(by_site),
                largest_site_fraction=float(counts.max()/counts.sum()),
                effective_sites=float(counts.sum()**2/(counts@counts)),
                site_residuals={s:median_interval(v) for s,v in sorted(by_site.items())})
    output=dict(groups=result,bindings={str(p):sha256(p) for p in (Path(__file__),mp,mr,rp,fp,vp,jp)},
        runtime_modified=False,promotion_allowed=False,
        limitation='Exact sample-site descriptive residual audit, no site feature or site correction fitted. Small selected support and disjoint fit/tune sites confound cohort and institution; not a causal root-cause identification.')
    out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

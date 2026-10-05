"""Bind source-fit BUN diagnostic to actual selected-epoch training draws."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_bounded_site_source import replacement_plan
from scripts.bounded_site_sampling import bounded_weights
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.audit_continued_selection_pairs import interval
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    rp=Path('artifacts/native_reconstruction_bun_dynamics_fit_20260920.json');dp=root/'bun_dynamics_fit_20260920.pt'
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt'
    paths=[root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    out=Path('artifacts/native_reconstruction_bun_fit_exposure_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    report=json.loads(rp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(dp)!=report['private_sha256']:raise ValueError('Rows changed')
    raw,ck,manifest=[torch.load(p,weights_only=False) for p in (dp,cp,mp)]
    joint,full,split=[torch.load(p,weights_only=False) for p in paths]
    columns=[joint['variables'].index(t) for t in joint['targets']];bun=joint['variables'].index('bun')
    shards=source_shards(joint,full,'expanded',split['roles'])
    stores={g:SourceStore([s for s in shards if s['group']==g and s['role']=='fit'],columns) for g in ('selected','broad')}
    groups={g:s.groups for g,s in stores.items()}
    weights={g:{c:bounded_weights([manifest['sample_sites'][g,'fit',k[0],c] for k in keys]) for c,keys in pool.items()} for g,pool in groups.items()}
    rng=np.random.default_rng(19);seen={g:{c:set() for c in columns} for g in groups};chosen=None
    for epoch in range(1,ck['epoch']+1):
        for group,key in replacement_plan(groups,weights,rng):seen[group][key[1]].add(key[0])
        expected=ck['log'][epoch-1]['unique_exposure']
        for g,pool in seen.items():
            for c,ids in pool.items():
                if len(ids)!=expected[g][joint['variables'][c]]:raise ValueError('Exposure count mismatch')
        if epoch==ck['selected_epoch']:chosen=deepcopy(seen)
    if rng.bit_generator.state!=ck['rng'] or seen!=ck['seen']:raise ValueError('Final sampler mismatch')
    result={}
    for g,rows in raw.items():
        result[g]={}
        for label,want in (('bun_seen',True),('bun_unseen',False)):
            selected=[r for r in rows if (r['patient'] in chosen[g][bun])==want]
            metrics={}
            for m in ('mae','smooth_l1'):
                errors={k:np.array([abs(r['predictions'][k]-r['truth']) for r in selected]) for k in ('full','no_body')}
                if m=='smooth_l1':errors={k:smooth_l1_from_error(v) for k,v in errors.items()}
                metrics[m]=interval(errors['no_body']-errors['full']) if len(selected)>=30 else dict(patients=len(selected),status='insufficient_support')
            result[g][label]=metrics
    out.write_text(json.dumps(dict(groups=result,selected_epoch=ck['selected_epoch'],sampler_exact=True,
        bindings={str(p):sha256(p) for p in (Path(__file__),rp,dp,cp,mp,*paths)},runtime_modified=False,promotion_allowed=False,
        limitation='In-sample descriptive errors. Unseen BUN pairs may share patients used for other targets. Source-fit population, not fresh generalization.'),indent=2)+'\n')
    print(result)


if __name__=='__main__':main()

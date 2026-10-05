"""Reconcile patient, expected sampler and realized draw-weighted BUN losses."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scripts.train_bounded_site_source import replacement_plan
from scripts.bounded_site_sampling import bounded_weights
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    rp=Path('artifacts/native_reconstruction_bun_dynamics_fit_20260920.json');dp=root/'bun_dynamics_fit_20260920.pt'
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt'
    paths=[root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    out=Path('artifacts/native_reconstruction_bun_weighted_objective_20260920.json')
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
    rng=np.random.default_rng(19);seen={g:{c:set() for c in columns} for g in groups};counts={g:Counter() for g in groups}
    for epoch in range(1,ck['epoch']+1):
        for group,key in replacement_plan(groups,weights,rng):
            seen[group][key[1]].add(key[0])
            if epoch<=ck['selected_epoch'] and key[1]==bun:counts[group][key[0]]+=1
    if rng.bit_generator.state!=ck['rng'] or seen!=ck['seen']:raise ValueError('Sampler mismatch')
    result={}
    for g,rows in raw.items():
        indexed={r['patient']:r for r in rows};keys=groups[g][bun]
        if set(indexed)!={k[0] for k in keys}:raise ValueError('Pairs changed')
        draw=np.array([counts[g][k[0]] for k in keys],dtype=float)
        if int(draw.sum())!=536*ck['selected_epoch']:raise ValueError('Draw mismatch')
        per={}
        for metric in ('mae','smooth_l1'):
            errors={a:np.array([abs(indexed[k[0]]['predictions'][a]-indexed[k[0]]['truth']) for k in keys]) for a in ('full','no_body')}
            if metric=='smooth_l1':errors={a:smooth_l1_from_error(v) for a,v in errors.items()}
            per[metric]={name:{**{a:float(np.average(v,weights=w)) for a,v in errors.items()},
                              'no_body_minus_full':float(np.average(errors['no_body']-errors['full'],weights=w))}
                         for name,w in (('patient_equal',np.ones(len(keys))),('expected_sampler',weights[g][bun]),('realized_draws',draw))}
        result[g]=dict(patients=len(keys),draws=int(draw.sum()),metrics=per)
    out.write_text(json.dumps(dict(groups=result,sampler_exact=True,selected_epoch=ck['selected_epoch'],
        bindings={str(p):sha256(p) for p in (Path(__file__),rp,dp,cp,mp,*paths)},runtime_modified=False,promotion_allowed=False,
        limitation='Final selected checkpoint rescored under historical draw weights, not the historical losses at each optimization step. BUN component only, not whole multitarget objective or convergence proof.'),indent=2)+'\n')
    print(result)


if __name__=='__main__':main()

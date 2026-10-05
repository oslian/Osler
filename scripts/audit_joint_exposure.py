"""Exact sampler replay for per-example exposure, not a causal diagnosis."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scripts.train_matched_broad_reconstruction import fixed_schedule
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_joint_exposure_20260918.json')
    if out.exists():raise ValueError('Existing audit')
    rp=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    d=json.loads(rp.read_text())
    for p,h in d['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training changed')
    ip=Path('data/research_recovery/broad_source_20260917/joint_source_manifest.pt')
    index=torch.load(ip,weights_only=False);variables=index['variables']
    cols=[variables.index(t)for t in index['targets']]
    groups={g:{c:[]for c in cols}for g in ('selected','broad')}
    for s in index['shards']:
        if s['role']!='fit':continue
        p=Path(s['path'])
        if sha256(p)!=s['sha256']:raise ValueError('Changed shard')
        for c in torch.load(p,weights_only=False)['samples']:
            if c in cols:groups[s['group']][c].append((s['patient'],c))
    counts={g:Counter()for g in groups};rng=np.random.default_rng(19)
    for epoch in range(8):
        schedules={g:fixed_schedule(groups[g],rng)for g in groups}
        # Consume the exact group-interleaving RNG calls from training.
        for i in range(17688):
            for g in rng.permutation(list(groups)):counts[g][schedules[g][i]]+=1
    result={}
    for g in groups:
        result[g]={}
        for c,keys in groups[g].items():
            values=np.array([counts[g][k]for k in keys])
            if values.sum()!=536*8 or (values==0).any():raise ValueError('Exposure mismatch')
            result[g][variables[c]]=dict(unique_pairs=len(keys),draws=int(values.sum()),
                minimum=int(values.min()),maximum=int(values.max()),mean=float(values.mean()),
                unique_fraction=float(len(keys)/values.sum()))
    out.write_text(json.dumps(dict(groups=result,bindings={str(p):sha256(p)for p in (Path(__file__),rp,ip)},
        limitation='Repeated draws are not independent patients. Exposure association is not proof of overfitting cause.',
        runtime_modified=False),indent=2)+'\n')
    print('All66 group-target schedules verified; total draws',sum(sum(c.values())for c in counts.values()))


if __name__=='__main__':main()

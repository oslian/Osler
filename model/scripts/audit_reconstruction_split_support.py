"""Check reusable frozen patient examples against all existing partitions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    source=Path('artifacts/native_reconstruction_expanded_20260913.json')
    parent=json.loads(source.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed source')
    q=pd.read_parquet(root/'queries.parquet')
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
    observed={}
    for shard in index['shards']:
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(shard['path'],weights_only=False)
        if s['patient'] in observed:raise ValueError('duplicate patient')
        observed[s['patient']]=set(s['samples'])
    targets=parent['source_eligible'];results={};hospital=None;paths=[Path(__file__),source]
    for seed in (7,19,31,43,59,71,83):
        path=root/f'partition_{seed}.npz';paths.append(path);part=dict(np.load(path))
        roles={r:set(q.loc[part[r],'uniquepid']) for r in ('fit','tune','evaluation','hospital')}
        if sum(map(len,roles.values()))!=len(set.union(*roles.values())):raise ValueError('role overlap')
        if set.union(*roles.values())!=set(observed):raise ValueError('uncovered patients')
        same=hospital is None or hospital==roles['hospital'];hospital=roles['hospital'] if hospital is None else hospital
        counts={r:{v:sum(index['variables'].index(v) in observed[p] for p in people) for v in targets}
                for r,people in roles.items()}
        results[str(seed)]=dict(role_patients={r:len(p) for r,p in roles.items()},counts=counts,
            same_hospital_as_seed7=same,source_unsupported=[v for v in targets if min(counts['fit'][v],counts['tune'][v])<30])
    out=Path('artifacts/native_reconstruction_split_support_20260914.json')
    if out.exists():raise ValueError('existing evidence')
    out.write_text(json.dumps(dict(complete=True,fixed_targets=targets,seeds=results,
        limitation='Repeated splits of same cohort and same hospital holdout, not independent external studies. Cached anchor draw fixed before repartition.',
        runtime_modified=False,promotion_allowed=False,bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    for seed,item in results.items():print(seed,item['role_patients'],'source unsupported',item['source_unsupported'])


if __name__=='__main__':main()

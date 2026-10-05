"""Separate source checkpoint selection from readout selection by whole sites."""
import json
from pathlib import Path
import pandas as pd
import torch
from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256


def separate_tune(roles, components):
    groups={}
    for patient, role in roles.items():
        if patient not in components:raise ValueError('missing component')
        groups.setdefault(components[patient],{}).setdefault(role,[]).append(patient)
    if any(len(g)!=1 for g in groups.values()):raise ValueError('component crosses roles')
    tune={c:g['tune'] for c,g in groups.items() if 'tune' in g}
    if len(tune)<2:raise ValueError('two tuning components required')
    assigned=dict(roles);counts={'encoder_tune':0,'readout_tune':0}
    for c in sorted(tune,key=lambda c:(-len(tune[c]),c)):
        role=min(counts,key=counts.get)
        for p in tune[c]:assigned[p]=role
        counts[role]+=len(tune[c])
    return assigned


def main():
    out=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    saved=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    if out.exists() or saved.exists():raise ValueError('existing partition')
    pp=Path('artifacts/native_reconstruction_site_partition_20260917.json')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed parent partition')
    split=Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    old=torch.load(split,weights_only=False)
    roles=separate_tune(old['roles'],old['components'])
    root=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    qp,ip=root/'queries.parquet',root/'expanded_reconstruction_20260913/index.pt'
    q=pd.read_parquet(qp);index=torch.load(ip,weights_only=False)
    check=q.assign(role=q.uniquepid.map(roles))
    if check.role.isna().any() or (check.groupby('hospitalid').role.nunique()!=1).any():raise ValueError('site isolation failed')
    names=('fit','encoder_tune','readout_tune','evaluation','reserved')
    counts={t:dict.fromkeys(names,0) for t in parent['target_counts']}
    for shard in index['shards']:
        p=durable(shard['path'])
        if sha256(p)!=shard['sha256']:raise ValueError('changed shard')
        data=torch.load(p,weights_only=False)
        if data['patient']!=shard['patient']:raise ValueError('identity mismatch')
        for c in data['samples']:
            t=index['variables'][c]
            if t in counts:counts[t][roles[data['patient']]]+=1
    torch.save(dict(roles=roles,components=old['components']),saved)
    result=dict(complete=True,role_patients={r:sum(v==r for v in roles.values()) for r in names},
        role_sites={r:int(check.loc[check.role==r,'hospitalid'].nunique()) for r in names},
        target_counts=counts,all_four_roles_supported=[t for t,v in counts.items() if all(v[r]>=30 for r in names[:4])],
        encoder_supported=[t for t,v in counts.items() if v['fit']>=30 and v['encoder_tune']>=30],
        bindings={str(p):sha256(p) for p in (Path(__file__),pp,split,qp,ip,saved)},
        policy='Preserve old fit/evaluation/reserved; split only tune whole components largest-first to smaller role, deterministic tie encoder-first. No target counts/values/errors used.',
        limitation='Support/partition only. Requires new encoder training excluding readout_tune; old weights are not independent. Reused development, warm-start provenance uncertified.',
        runtime_modified=False,promotion_allowed=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['role_patients'],result['role_sites'])
    print('four-role support',len(result['all_four_roles_supported']))
    print('unsupported',{t:v for t,v in counts.items() if t not in result['all_four_roles_supported']})


if __name__=='__main__':main()

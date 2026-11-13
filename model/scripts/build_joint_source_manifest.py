"""Join source shard references, never union asynchronous patient histories."""
import json
from pathlib import Path
import torch
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_joint_source_manifest_20260917.json')
    private=Path('data/research_recovery/broad_source_20260917/joint_source_manifest.pt')
    if out.exists() or private.exists():raise ValueError('Existing manifest')
    split=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    oldindex=BACKUP/'map_native_anchor_support/expanded_reconstruction_20260913/index.pt'
    broadindex=Path('data/research_recovery/broad_source_20260917/matched_samples/index.pt')
    confirmindex=Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt')
    train=Path('artifacts/native_reconstruction_matched_broad_training_20260917.json')
    tr=json.loads(train.read_text())
    for p,h in tr['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed training input')
    roles=torch.load(split,weights_only=False)['roles']
    old=torch.load(oldindex,weights_only=False);broad=torch.load(broadindex,weights_only=False)
    if old['variables']!=broad['variables']:raise ValueError('Schema mismatch')
    variables=old['variables'];targets=tr['source_targets']
    held={p for p,r in roles.items() if r not in ('fit','encoder_tune')}
    held.update(s['patient']for s in torch.load(confirmindex,weights_only=False)['shards'])
    seen={};shards=[]
    counts={g:{r:{t:0 for t in targets}for r in ('fit','encoder_tune')}for g in ('selected','broad')}
    people={g:{r:0 for r in ('fit','encoder_tune')}for g in counts}
    for group,index in (('selected',old),('broad',broad)):
        for item in index['shards']:
            role=roles[item['patient']]if group=='selected'else item['role']
            if role not in ('fit','encoder_tune'):continue
            p=durable(item['path'])
            if sha256(p)!=item['sha256']:raise ValueError('Shard changed')
            s=torch.load(p,weights_only=False);patient=s['patient']
            if patient!=item['patient'] or patient in held or patient in seen:raise ValueError('Patient leakage or duplicate')
            seen[patient]=(group,role);people[group][role]+=1
            for c,e in s['samples'].items():
                t=variables[c]
                if t in targets:counts[group][role][t]+=1
            shards.append(dict(patient=patient,group=group,role=role,path=str(p),sha256=item['sha256']))
    for group,p in people.items():
        if p!={'fit':536,'encoder_tune':90}:raise ValueError('Unmatched groups')
    torch.save(dict(shards=shards,variables=variables,targets=targets),private)
    paths=[Path(__file__),split,oldindex,broadindex,confirmindex,train,private]
    out.write_text(json.dumps(dict(complete=True,people=people,support=counts,
        all_group_role_supported=[t for t in targets if all(counts[g][r][t]>=30 for g in counts for r in counts[g])],
        bindings={str(p):sha256(p)for p in paths},runtime_modified=False,
        limitation='Union of disjoint source patients only; original per-patient event histories unchanged. Preparation, not model efficacy.'),indent=2)+'\n')
    print('Joint source verified',people,flush=True)


if __name__=='__main__':main()

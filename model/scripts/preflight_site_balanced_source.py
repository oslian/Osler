"""Verify complete source-site mapping and repetition cost before training."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import torch
from scripts.reconstruction_source_store import source_shards
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    out=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    dest=root/'site_balance_manifest_20260920.pt'
    if out.exists() or dest.exists():raise ValueError('Existing preflight')
    paths=[root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    joint,full,split=[torch.load(p,weights_only=False) for p in paths]
    sidecars={}
    for group,name in (('selected','selected_sample_sites'),('broad','sample_sites')):
        p=root/f'{name}_20260920.pt';rp=Path(f'artifacts/native_reconstruction_{"selected_sites" if group=="selected" else "sample_sites"}_20260920.json')
        report=json.loads(rp.read_text())
        if sha256(p)!=report['private_sha256']:raise ValueError('Sidecar changed')
        sidecars[group]=torch.load(p,weights_only=False);paths.extend([p,rp])
    # Full encoder role retained for preflight; no changed checkpoint selection.
    shards=source_shards(joint,full,'expanded',split['roles'])
    shards=[s for s in shards if not(s['group']=='broad' and s['role']=='encoder_tune')]
    shards += [dict(s,group='broad') for s in full['shards'] if s['role']=='encoder_tune']
    if len({s['patient'] for s in shards})!=len(shards):raise ValueError('Patient overlap')
    counts=defaultdict(Counter);mapping={};patients=defaultdict(set)
    for s in shards:
        p=Path(s['path'])
        if sha256(p)!=s['sha256']:raise ValueError('Sample changed')
        sample=torch.load(p,weights_only=False)
        if sample['patient']!=s['patient']:raise ValueError('Patient mismatch')
        sites=sidecars[s['group']];patient=s['patient']
        for c in sample['samples']:
            site=sites['single_site_patients'].get(patient)
            if site is None:site=sites['ambiguous_patient_samples'][patient][c]['site']
            key=(s['group'],s['role'],patient,c)
            mapping[key]=site
            counts[s['group'],s['role'],joint['variables'][c]][site]+=1
        patients[s['group'],s['role']].add(patient)
    summaries={}
    for (group,role,target),values in counts.items():
        n=sum(values.values());k=len(values);minimum=min(values.values())
        summaries.setdefault(group,{}).setdefault(role,{})[target]=dict(pairs=n,sites=k,min_site_pairs=minimum,max_site_pairs=max(values.values()),
            maximum_equal_site_patient_weight_multiplier=n/(k*minimum),trained=target in joint['targets'])
    bindings={str(p):sha256(p) for p in [*paths,Path(__file__)]}
    torch.save(dict(sample_sites=mapping,shards=shards,bindings=bindings),dest)
    result=dict(groups=summaries,source_patients={g:{r:len(patients[g,r]) for r in ('fit','encoder_tune')} for g in ('selected','broad')},
        pairs=len(mapping),bindings=bindings,private_sha256=sha256(dest),runtime_modified=False,promotion_allowed=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('Patients',result['source_patients'],'pairs',len(mapping))
    for g in ('selected','broad'):
        items=summaries[g]['fit']
        print(g,'largest equal-site upweights',sorted(((t,r['maximum_equal_site_patient_weight_multiplier'],r['min_site_pairs']) for t,r in items.items()),key=lambda x:x[1],reverse=True)[:5])


if __name__=='__main__':main()

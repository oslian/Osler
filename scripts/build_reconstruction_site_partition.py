"""Freeze one common support-blind hospital-component development partition."""
import json
from pathlib import Path
import pandas as pd
import torch
from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.site_partitions import site_partition


def main():
    root = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    qp, ip = root/'queries.parquet', root/'expanded_reconstruction_20260913/index.pt'
    report_path = Path('artifacts/native_reconstruction_expanded_20260913.json')
    parent = json.loads(report_path.read_text())
    for p in (qp, ip):
        if [h for f,h in parent['bindings'].items() if Path(f).name==p.name] != [sha256(p)]:
            raise ValueError('changed original input')
    out = Path('artifacts/native_reconstruction_site_partition_20260917.json')
    saved = Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    if out.exists() or saved.exists():
        raise ValueError('existing partition')
    q = pd.read_parquet(qp)
    index = torch.load(ip,weights_only=False)
    roles, components = site_partition(q, index['roles']['hospital'])
    counts = {t:dict.fromkeys(('fit','tune','evaluation','reserved'),0) for t in parent['source_eligible']}
    for shard in index['shards']:
        p = durable(shard['path'])
        if sha256(p)!=shard['sha256']:
            raise ValueError('changed shard')
        s = torch.load(p,weights_only=False)
        if s['patient']!=shard['patient']:
            raise ValueError('shard identity mismatch')
        for c in s['samples']:
            t = index['variables'][c]
            if t in counts:
                counts[t][roles[s['patient']]] += 1
    eligible = [t for t,v in counts.items() if v['fit']>=30 and v['tune']>=30]
    complete = [t for t in eligible if counts[t]['evaluation']>=30]
    torch.save(dict(roles=roles,components=components,original_hospital_patients=index['roles']['hospital']),saved)
    paths = [Path(__file__),Path('whole_body_jepa/site_partitions.py'),qp,ip,report_path,saved]
    report = dict(complete=True,role_patients={r:sum(v==r for v in roles.values()) for r in set(roles.values())},
        role_sites={r:int(q.loc[q.uniquepid.map(roles).eq(r),'hospitalid'].nunique()) for r in set(roles.values())},
        target_counts=counts,source_eligible=eligible,three_role_supported=complete,
        patient_and_site_disjoint=True,runtime_modified=False,promotion_allowed=False,
        policy='Common largest-component-first 60/20/20 patient-deficit allocation; no labels or errors used for assignment. Reserve all components touched by original hospital patients.',
        limitation='New development partition only, not fresh lockbox or trained efficacy. Warm-start pretraining independence remains uncertified.',
        bindings={str(p):sha256(p) for p in paths})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['role_patients'],report['role_sites'])
    print('source eligible',len(eligible),'three-role supported',len(complete))
    print('unsupported',{t:v for t,v in counts.items() if t not in complete})


if __name__=='__main__':main()

"""Outcome-blind, component-disjoint fit/selection/calibration/test roles."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch

from scripts.preflight_encoder_crossfit import components
from whole_body_jepa.multisource_intervals import sha256

ROLES = ('fit', 'selection', 'calibration', 'test')


def partition(patient_sites):
    mapping = components(patient_sites)
    people = np.array(sorted(mapping))
    groups = np.array([mapping[p] for p in people])
    if len(set(groups)) < 12:
        raise ValueError('At least twelve independent components required')
    folds = []
    for development, test in GroupKFold(5).split(people, groups=groups):
        remainder, calibration = next(GroupKFold(4).split(development, groups=groups[development]))
        remaining = development[remainder]
        fit, selection = next(GroupKFold(3).split(remaining, groups=groups[remaining]))
        assigned = {p: 'fit' for p in people[remaining[fit]]}
        assigned.update({p:'selection' for p in people[remaining[selection]]})
        assigned.update({p:'calibration' for p in people[development[calibration]]})
        assigned.update({p:'test' for p in people[test]})
        if len(assigned)!=len(people):raise ValueError('Lost partition identity')
        site_roles = {}
        for p,r in assigned.items():
            for site in patient_sites[p]:site_roles.setdefault(site,set()).add(r)
        if any(len(r)!=1 for r in site_roles.values()):raise ValueError('Hospital crosses roles')
        folds.append(assigned)
    if any(sum(f[p]=='test' for f in folds)!=1 for p in people):
        raise ValueError('Outer test coverage not exactly once')
    return folds


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    out=Path('artifacts/native_reconstruction_additional_domain_partitions_20260922.json')
    private=root/'partitions_20260922.pt'
    if out.exists() or private.exists():raise ValueError('Existing partition')
    rp=Path('artifacts/native_reconstruction_additional_masked_history_full_20260921.json')
    report=json.loads(rp.read_text())
    for path,h in report['bindings'].items():
        if sha256(Path(path))!=h:raise ValueError('Changed masked-history dependency')
    pp=root/'provenance.parquet'; prov=pd.read_parquet(pp)
    # Include ALL candidate people, including those without accepted examples.
    patient_sites=prov.groupby('uniquepid').hospitalid.agg(set).to_dict()
    expected=prov.groupby('uniquepid').component.agg(set)
    actual=components(patient_sites)
    if any(expected[p]!={actual[p]} for p in actual):raise ValueError('Component provenance mismatch')
    folds=partition(patient_sites)
    ip=root/'masked_history_full/index.pt'
    if sha256(ip)!=report['private_index_sha256']:raise ValueError('Changed acceptance index')
    shards=torch.load(ip,weights_only=False); parts=[]
    for item in shards:
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Changed decision shard')
        parts.append(pd.read_parquet(item['path'],columns=['uniquepid','variable','component','status']))
    frame=pd.concat(parts,ignore_index=True)
    if frame.duplicated(['uniquepid','variable']).any():raise ValueError('Duplicate anchor')
    if frame.status.value_counts().to_dict()!=report['summary']['rejection_counts']:
        raise ValueError('Acceptance count mismatch')
    targets=sorted(frame.variable.unique()); summaries=[]
    for number,roles in enumerate(folds):
        accepted=frame.loc[frame.status.eq('accepted')].copy()
        accepted['role']=accepted.uniquepid.map(roles)
        if accepted.role.isna().any():raise ValueError('Unassigned example')
        counts={}
        for target in targets:
            counts[target]={}
            for role in ROLES:
                part=accepted.loc[accepted.variable.eq(target)&accepted.role.eq(role)]
                n=part.groupby('component').uniquepid.nunique()
                counts[target][role]=dict(patients=int(n.sum()),components=len(n),
                                         components_at_least30=int((n>=30).sum()))
        summaries.append(dict(fold=number,
            patients={r:sum(v==r for v in roles.values()) for r in ROLES},
            components={r:len({actual[p] for p,v in roles.items() if v==r}) for r in ROLES},
            target_support=counts))
    paths=[Path(__file__),rp,pp,ip,Path('scripts/preflight_encoder_crossfit.py')]
    bindings={str(p):sha256(p) for p in paths}
    torch.save(dict(folds=folds,roles=ROLES,bindings=bindings),private)
    supported=[t for t in targets if all(f['target_support'][t][r]['components_at_least30']>=3
               for f in summaries for r in ROLES)]
    result=dict(candidate_people=len(patient_sites),components=len(set(actual.values())),
        folds=summaries,all_role_all_fold_three_domain_support=supported,
        bindings=bindings,private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        protocol='Fixed GroupKFold(5) outer; first GroupKFold(4) development fold calibration; first GroupKFold(3) remainder fold selection; remaining fit. Roles depend only on full patient-site identities/counts. Calibration cannot select models or train encoder; test cannot select anything. No retries based on target counts/errors.',
        limitation='Prospective roles inside already inspected eICU development, not a new untouched external lockbox. >=30patients in >=3components per role is a support feasibility condition, not a promotion gate. Selected insulin cohort mismatch persists. Fit-only input vocabulary/normalization and training remain pending.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidate_people=len(patient_sites),folds=[{k:v for k,v in f.items() if k!='target_support'} for f in summaries],supported_targets=supported,unsupported_targets=sorted(set(targets)-set(supported))),indent=2),flush=True)


if __name__=='__main__':main()

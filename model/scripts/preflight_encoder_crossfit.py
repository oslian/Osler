"""Plan encoder-independent folds using complete patient/site components.

This does not train an encoder or certify inherited checkpoint provenance.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch

from whole_body_jepa.multisource_intervals import sha256


def components(patient_sites):
    parent = {}

    def find(site):
        parent.setdefault(site, site)
        while parent[site] != site:
            parent[site] = parent[parent[site]]
            site = parent[site]
        return site

    for patient in sorted(patient_sites):
        sites = sorted(patient_sites[patient])
        if not sites:
            raise ValueError('Patient without full-history site provenance')
        for site in sites[1:]:
            a, b = find(sites[0]), find(site)
            parent[max(a, b)] = min(a, b)
        find(sites[0])
    return {p: find(min(sites)) for p, sites in patient_sites.items()}


def make_folds(patient_sites):
    """No measurements, outcomes, target support or model scores select roles."""
    mapping = components(patient_sites)
    people = np.array(sorted(mapping))
    groups = np.array([mapping[p] for p in people])
    if len(set(groups)) < 5:
        raise ValueError('Five independent hospital components required')
    folds = []
    for development, test in GroupKFold(5).split(people, groups=groups):
        inner_groups = groups[development]
        if len(set(inner_groups)) < 2:
            raise ValueError('No independent inner selection sites')
        fit, tune = next(GroupKFold(min(4, len(set(inner_groups)))).split(
            people[development], groups=inner_groups))
        roles = {p: 'fit' for p in people[development[fit]]}
        roles.update({p: 'encoder_tune' for p in people[development[tune]]})
        roles.update({p: 'test' for p in people[test]})
        site_roles = {}
        for person, role in roles.items():
            for site in patient_sites[person]:
                site_roles.setdefault(site, set()).add(role)
        if any(len(r) != 1 for r in site_roles.values()):
            raise ValueError('Hospital crosses fold roles')
        folds.append(roles)
    if any(sum(f[p] == 'test' for f in folds) != 1 for p in people):
        raise ValueError('Outer coverage is not exactly once per person')
    return folds


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    mp = root / 'site_balance_manifest_20260920.pt'
    pp = Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    output = Path('artifacts/native_reconstruction_encoder_crossfit_preflight_20260920.json')
    private = root / 'encoder_crossfit_folds_20260920.pt'
    if output.exists() or private.exists():
        raise ValueError('Existing crossfit evidence')
    if sha256(mp) != json.loads(pp.read_text())['private_sha256']:
        raise ValueError('Changed source manifest')
    manifest = torch.load(mp, weights_only=False)
    for path, digest in manifest['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed source provenance')
    # Only former fit patients enter cross-fitting. Prior selection remains out.
    shards = [s for s in manifest['shards'] if s['role'] == 'fit']
    people = {s['patient'] for s in shards}
    if len(people) != len(shards):
        raise ValueError('Duplicate source patient')
    raw_people = pd.read_csv(raw, usecols=['uniquepid', 'hospitalid'], dtype=str)
    patient_sites = raw_people.loc[raw_people.uniquepid.isin(people)].groupby('uniquepid').hospitalid.agg(set).to_dict()
    if set(patient_sites) != people:
        raise ValueError('Patient identity missing from raw crosswalk')
    for (group, role, person, col), site in manifest['sample_sites'].items():
        if role == 'fit' and str(site) not in patient_sites[person]:
            raise ValueError('Sample site disagrees with complete patient history')
    folds = make_folds(patient_sites)
    joint = root / 'joint_source_manifest.pt'
    schema = torch.load(joint, weights_only=False)
    targets = schema['targets']; columns = {schema['variables'].index(t): t for t in targets}
    summaries = []
    for number, roles in enumerate(folds):
        counts = {t: {r: 0 for r in ('fit', 'encoder_tune', 'test')} for t in targets}
        target_sites = {t: {r: set() for r in ('fit', 'encoder_tune', 'test')} for t in targets}
        for (group, role, person, col), site in manifest['sample_sites'].items():
            if role == 'fit' and col in columns:
                t, r = columns[col], roles[person]
                counts[t][r] += 1
                target_sites[t][r].add(str(site))
        summaries.append(dict(fold=number, patients=dict(Counter(roles.values())),
            sites={r: len(set().union(*(patient_sites[p] for p in roles if roles[p] == r))) for r in ('fit', 'encoder_tune', 'test')},
            targets={t: dict(patients=counts[t], sites={r: len(s) for r, s in target_sites[t].items()}) for t in targets}))
    bindings = {str(p): sha256(p) for p in (Path(__file__), mp, pp, raw, joint)}
    torch.save(dict(folds=folds, shards=shards, bindings=bindings), private)
    result = dict(complete=True, source_patients=len(people), source_sites=len(set().union(*patient_sites.values())),
        hospital_components=len(set(components(patient_sites).values())), multi_site_patients=sum(len(s)>1 for s in patient_sites.values()),
        targets=len(targets), folds=summaries,
        all_fold_supported=[t for t in targets if all(all(f['targets'][t]['patients'][r]>=30 for r in ('fit','encoder_tune','test')) for f in summaries)],
        bindings=bindings, private_sha256=sha256(private), runtime_modified=False, promotion_allowed=False,
        limitation='Partition/support only; no encoder training. Cached body features and bounded7 weights cannot establish outer independence. Inherited preprocessing/pretraining requires separate provenance or fold-local reconstruction. Reused source development, not a new external lockbox.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds','bindings','private_sha256')}, indent=2))
    print('Fold role counts', [f['patients'] for f in summaries])


if __name__ == '__main__':
    main()

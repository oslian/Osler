"""Outcome-blind feasibility of multi-component source calibration."""
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.preflight_encoder_crossfit import components
from whole_body_jepa.multisource_intervals import sha256


def partitions(available, minimum=3):
    available = tuple(sorted(available))
    return [(set(c), set(available)-set(c))
            for size in range(minimum, len(available)-minimum+1)
            for c in combinations(available, size)]


def supported_domains(counts, domains, minimum_patients=30):
    return int(sum(counts.get(c, 0) >= minimum_patients for c in domains))


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    mp = root/'site_balance_manifest_20260920.pt'
    jp = root/'joint_source_manifest.pt'
    rp = Path('artifacts/native_reconstruction_encoder_crossfit_preflight_20260920.json')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    out = Path('artifacts/native_reconstruction_calibration_domain_support_20260921.json')
    if out.exists():
        raise ValueError('Existing feasibility report')
    previous = json.loads(rp.read_text())
    if sha256(fp) != previous['private_sha256']:
        raise ValueError('Changed fold plan')
    for p,h in previous['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed source dependency')
    plan = torch.load(fp, weights_only=False)
    manifest = torch.load(mp, weights_only=False)
    schema = torch.load(jp, weights_only=False)
    columns = {schema['variables'].index(t):t for t in schema['targets']}
    people = {s['patient'] for s in plan['shards']}
    table = pd.read_csv(raw, usecols=['uniquepid', 'hospitalid'], dtype=str)
    sites = table.loc[table.uniquepid.isin(people)].groupby('uniquepid').hospitalid.agg(set).to_dict()
    if set(sites) != people:
        raise ValueError('Missing raw provenance')
    component = components(sites); all_components = set(component.values())
    counts = {t:{g:{} for g in ('selected','broad')} for t in columns.values()}
    seen = set()
    for (g, role, person, col), site in manifest['sample_sites'].items():
        if role != 'fit' or person not in people or col not in columns:
            continue
        key = (person, col)
        if key in seen:
            raise ValueError('Duplicate patient-target support')
        seen.add(key)
        c = component[person]; d = counts[columns[col]][g]
        d[c] = d.get(c, 0)+1
    overall = {t:{g:dict(patients=sum(d.values()), components_with_any=len(d),
                        components_with_30=supported_domains(d,all_components))
                 for g,d in gs.items()} for t,gs in counts.items()}
    reports = []
    for fold,roles in enumerate(plan['folds']):
        test = {component[p] for p,r in roles.items() if r=='test'}
        existing = {component[p] for p,r in roles.items() if r=='encoder_tune'}
        choices = partitions(all_components-test)
        targets = {}
        for t,gs in counts.items():
            feasible = 0; best_min = 0
            for cal, fit in choices:
                support = [supported_domains(gs[g], role) for g in gs for role in (cal, fit)]
                best_min = max(best_min,min(support))
                feasible += min(support)>=3
            targets[t] = dict(feasible_partitions=int(feasible),
                              best_minimum_supported_components=best_min)
        reports.append(dict(fold=fold, outer_components=len(test),
            existing_calibration_components=len(existing),
            development_components=len(all_components-test),
            minimum_three_per_role_choices=len(choices), targets=targets))
    summary = dict(patients=len(people), hospitals=len(set().union(*sites.values())),
        components=len(all_components), targets=len(counts),
        targets_with_six_supported_components_both_groups=[t for t,r in overall.items()
            if all(v['components_with_30']>=6 for v in r.values())],
        feasible_target_fold_pairs=sum(v['feasible_partitions']>0 for r in reports for v in r['targets'].values()))
    out.write_text(json.dumps(dict(summary=summary, overall_support=overall, folds=reports,
        bindings={str(p):sha256(p) for p in [Path(__file__),fp,mp,jp,rp,raw]},
        runtime_modified=False,promotion_allowed=False,
        limitation='Support feasibility, not efficacy. Proposed diagnostic needs at least three calibration and three fit hospital components with >=30 patients per target/group in each. This is not an existing promotion threshold and is not chosen using prediction errors. All development-component subsets leaving at least three components per role are enumerated.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Role components',[(r['outer_components'],r['existing_calibration_components'],r['minimum_three_per_role_choices']) for r in reports])


if __name__=='__main__':
    main()

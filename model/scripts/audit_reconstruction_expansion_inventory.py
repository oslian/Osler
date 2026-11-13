"""Count expansion candidates using full stay provenance, without lab outcomes."""
import json
from pathlib import Path
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_reconstruction_expansion_inventory_20260917.json')
    if output.exists():
        raise ValueError('Existing inventory')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    qp = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    sp = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    ancestry = Path('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet')
    p = pd.read_csv(raw, usecols=['patientunitstayid','uniquepid','hospitalid'], dtype=str)
    if p.isna().any().any() or p.patientunitstayid.duplicated().any():
        raise ValueError('Invalid patient crosswalk')
    q = pd.read_parquet(qp, columns=['uniquepid','hospitalid'])
    roles = torch.load(sp, weights_only=False)['roles']
    hospital_roles = {}
    for row in q.itertuples():
        site = str(int(row.hospitalid)); role = roles[row.uniquepid]
        if site in hospital_roles and hospital_roles[site] != role:
            raise ValueError('Existing query sites cross roles')
        hospital_roles[site] = role
    a = pd.read_parquet(ancestry, columns=['subject_id','hospitalid'])
    ancestral_people = set(a.subject_id.astype(str))
    ancestral_sites = set(a.hospitalid.astype(int).astype(str))
    existing = set(roles)
    histories = p.groupby('uniquepid').hospitalid.agg(set)
    counts = {r: dict(new_patients_with_full_history_inside_role=0, new_stays=0) for r in sorted(set(roles.values()))}
    sizes = p.groupby('uniquepid').size()
    excluded = dict(existing=0, ancestry=0, crosses_roles_or_unassigned_sites=0, unassigned_sites_only=0)
    prior_crossings = 0
    for patient, sites in histories.items():
        mapped = {hospital_roles.get(s, 'unassigned') for s in sites}
        if patient in existing:
            excluded['existing'] += 1
            if mapped != {roles[patient]}:
                prior_crossings += 1
            continue
        if patient in ancestral_people or sites & ancestral_sites:
            excluded['ancestry'] += 1
            continue
        if mapped == {'unassigned'}:
            excluded['unassigned_sites_only'] += 1
        elif len(mapped) != 1 or 'unassigned' in mapped:
            excluded['crosses_roles_or_unassigned_sites'] += 1
        else:
            role = next(iter(mapped))
            counts[role]['new_patients_with_full_history_inside_role'] += 1
            counts[role]['new_stays'] += int(sizes[patient])
    payload = dict(complete=True, raw_patients=int(p.uniquepid.nunique()), raw_stays=len(p), role_candidates=counts,
        exclusions=excluded, existing_patients_with_additional_out_of_role_sites=prior_crossings,
        bindings={str(f):sha256(f) for f in (Path(__file__),raw,qp,sp,ancestry)},
        limitation='Provenance-only upper bound: no analyte availability, result-entry lag, masks, quality or target label eligibility checked. Candidates are not assigned or trained; no untouched-data claim.',
        runtime_modified=False,promotion_allowed=False)
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('bindings','limitation')},indent=2))


if __name__ == '__main__':
    main()

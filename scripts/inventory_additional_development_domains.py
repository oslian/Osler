"""Provenance-only new source domains, excluding connected known domains."""
import json
from pathlib import Path
import pandas as pd
import torch

from scripts.preflight_encoder_crossfit import components
from whole_body_jepa.multisource_intervals import sha256


def disconnected_candidates(histories, forbidden_sites, forbidden_people):
    mapping = components(histories)
    blocked = {mapping[p] for p, sites in histories.items()
               if sites & forbidden_sites or p in forbidden_people}
    candidates = {p for p,c in mapping.items() if c not in blocked}
    return candidates, mapping


def main():
    root = Path('data/research_recovery')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    qp = root/'rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet'
    sp = root/'reconstruction_nested_selection_20260917.pt'
    fp = root/'broad_source_20260917/encoder_crossfit_folds_20260920.pt'
    bp = root/'broad_source_20260917/provenance.parquet'
    cp = root/'expanded_confirmation_20260917/provenance.parquet'
    ancestor = Path('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet')
    ap = Path('artifacts/native_reconstruction_ancestry_identity_20260917.json')
    out = Path('artifacts/native_reconstruction_additional_development_domains_20260921.json')
    private = root/'additional_development_domains_20260921'
    if out.exists() or private.exists():
        raise ValueError('Existing inventory')
    audit = json.loads(ap.read_text())
    for p,h in audit['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed ancestry identity evidence')
    patients = pd.read_csv(raw, usecols=['patientunitstayid','uniquepid','hospitalid'],dtype=str)
    if patients.isna().any().any() or patients.patientunitstayid.duplicated().any():
        raise ValueError('Invalid raw crosswalk')
    histories = patients.groupby('uniquepid').hospitalid.agg(set).to_dict()
    roles = torch.load(sp,weights_only=False)['roles']
    folds = torch.load(fp,weights_only=False)
    q = pd.read_parquet(qp,columns=['uniquepid','hospitalid'])
    a = pd.read_parquet(ancestor,columns=['subject_id','hospitalid'])
    broad = pd.read_parquet(bp,columns=['uniquepid','hospitalid'])
    confirmation = pd.read_parquet(cp,columns=['uniquepid','hospitalid'])
    known_people = set(roles)|{s['patient'] for s in folds['shards']}|set(q.uniquepid)|set(a.subject_id.astype(str))|set(broad.uniquepid)|set(confirmation.uniquepid)
    if not known_people.issubset(histories):
        raise ValueError('Unknown patient identity')
    known_sites = set().union(*(histories[p] for p in known_people))
    # Full-table connected components exclude transitive cross-hospital bridges,
    # including people not present in the old bounded research cohort.
    candidates, mapping = disconnected_candidates(histories,known_sites,known_people)
    result = patients.loc[patients.uniquepid.isin(candidates)].copy()
    result['component'] = result.uniquepid.map(mapping)
    if set(result.hospitalid)&known_sites or set(result.uniquepid)&known_people:
        raise ValueError('Known-domain leakage')
    ordinal = {c:i for i,c in enumerate(sorted(result.component.unique()))}
    groups=[]
    for c, part in result.groupby('component'):
        groups.append(dict(component_ordinal=ordinal[c],patients=int(part.uniquepid.nunique()),
                           stays=len(part),hospitals=int(part.hospitalid.nunique())))
    direct = {p for p,s in histories.items() if p not in known_people and not s&known_sites}
    summary = dict(raw_patients=len(histories),raw_hospitals=int(patients.hospitalid.nunique()),
        raw_components=len(set(mapping.values())),known_patients=len(known_people),known_hospitals=len(known_sites),
        direct_disjoint_patients=len(direct),transitive_bridge_excluded=len(direct-candidates),
        candidate_patients=len(candidates),candidate_stays=len(result),
        candidate_hospitals=int(result.hospitalid.nunique()),candidate_components=len(groups))
    private.mkdir();path=private/'provenance.parquet';result.to_parquet(path,index=False)
    dependencies=[Path(__file__),raw,qp,sp,fp,bp,cp,ancestor,ap,Path('scripts/preflight_encoder_crossfit.py')]
    out.write_text(json.dumps(dict(summary=summary,components=groups,
        private_sha256=sha256(path),bindings={str(p):sha256(p) for p in dependencies},
        runtime_modified=False,promotion_allowed=False,
        limitation='Unassigned candidate inventory only, no role assignment, extraction or model fitting. Excludes all known roles/current ancestry and their complete-history connected domains for this pipeline. No claim of never used anywhere in historical project, clinical cohort eligibility, valid measurement support or untouched external lockbox.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

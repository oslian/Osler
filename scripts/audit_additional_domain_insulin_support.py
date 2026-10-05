"""Retrospective cohort-support ceiling; never an inference-time phenotype."""
import json
from pathlib import Path

import pandas as pd

from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.multisource_intervals import sha256


def rate_change_stays(rates):
    changes = []
    for (stay, label), group in rates.groupby(['stay_id', 'label']):
        conflict = group.groupby('offset').value.transform('nunique').gt(1)
        clean = group.loc[~conflict].sort_values('offset').drop_duplicates('offset')
        gap = clean.offset.diff()
        valid = gap.gt(0) & gap.le(360) & clean.value.diff().abs().gt(1e-8)
        changes.extend((str(stay), float(offset)) for offset in clean.loc[valid, 'offset'])
    frame = pd.DataFrame(changes, columns=['patientunitstayid', 'offset'])
    # Match the historical exclusion of simultaneous changes in multiple labels.
    return set(frame.loc[~frame.duplicated(['patientunitstayid', 'offset'], keep=False),
                         'patientunitstayid'])


def support_counts(support, members):
    result = {}
    for target, part in support.groupby('variable'):
        count = part.loc[part.uniquepid.isin(members)].groupby('component').uniquepid.nunique()
        result[target] = dict(patients=int(count.sum()), components=int(len(count)),
                              components_at_least30=int((count>=30).sum()))
    return result


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    output = Path('artifacts/native_reconstruction_additional_insulin_support_20260921.json')
    if output.exists():
        raise ValueError('Completed audit already exists')
    pp = root/'provenance.parquet'; provenance = pd.read_parquet(pp)
    paths = [Path(__file__), pp, Path('scripts/audit_eicu_insulin_sequence_inventory.py'),
             Path('whole_body_jepa/event_treatment.py'), Path('scripts/build_rate_anchored_potassium.py'),
             Path('scripts/build_insulin_potassium_pairs.py'), Path('scripts/audit_map_native_anchor_support.py')]
    parts = []
    for kind in ('labs', 'bedside'):
        rp = Path(f'artifacts/native_reconstruction_additional_domain_{kind}_20260921.json')
        report = json.loads(rp.read_text())
        if report['bindings'][str(pp)] != sha256(pp):
            raise ValueError('Changed event provenance')
        sp = root/kind/'patient_target_support.parquet'
        if sha256(sp) != report['private_support_sha256']:
            raise ValueError('Changed support table')
        parts.append(pd.read_parquet(sp)); paths.extend([rp, sp])
    support = pd.concat(parts, ignore_index=True).drop_duplicates()
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    paths.append(raw)
    rates = read_insulin_rates(raw)
    rates['stay_id'] = rates.stay_id.astype(str)
    rates = rates.loc[rates.stay_id.isin(provenance.patientunitstayid)]
    any_stays = set(rates.stay_id)
    change_stays = rate_change_stays(rates)
    any_people = set(provenance.loc[provenance.patientunitstayid.isin(any_stays), 'uniquepid'])
    change_people = set(provenance.loc[provenance.patientunitstayid.isin(change_stays), 'uniquepid'])
    if not change_people <= any_people:
        raise ValueError('Invalid nested membership')
    flags = provenance.copy()
    flags['documented_insulin_patient'] = flags.uniquepid.isin(any_people)
    flags['documented_rate_change_patient'] = flags.uniquepid.isin(change_people)
    private = root/'retrospective_insulin_support.parquet'; flags.to_parquet(private, index=False)
    summary = dict(candidate_patients=int(provenance.uniquepid.nunique()),
        compatible_rate_events=len(rates), patients_with_documented_insulin=len(any_people),
        patients_with_documented_rate_change=len(change_people),
        hospitals_with_rate_change=int(flags.loc[flags.documented_rate_change_patient, 'hospitalid'].nunique()),
        components_with_rate_change=int(flags.loc[flags.documented_rate_change_patient, 'component'].nunique()))
    by_group = {name: support_counts(support, people) for name, people in
                [('documented_insulin', any_people), ('documented_rate_change', change_people)]}
    output.write_text(json.dumps(dict(summary=summary, target_support=by_group,
        bindings={str(p): sha256(p) for p in paths}, private_sha256=sha256(private),
        inference_feature_allowed=False, selected_cohort_equivalence_established=False,
        runtime_modified=False, promotion_allowed=False,
        limitation='Whole-stay retrospective upper-bound support only. Historical selected cohort additionally required potassium future-pair availability and restricted stay/role/MAP support. Neither insulin documentation nor absence establishes diabetes or untreated status. Rate availability is not verified, and some records occur after a reconstruction anchor. Do not use these flags for routing or claim they recreate the old selected population.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps({g: {t: r[t] for t in ('creatinine','bun','glucose','albumin','total_protein')}
                      for g, r in by_group.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()

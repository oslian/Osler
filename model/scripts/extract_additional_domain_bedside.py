"""Availability-qualified nursing vitals for unassigned development domains."""
import json
from pathlib import Path

import duckdb
import pandas as pd
import torch

from extractors.eicu_sepsis_transition_extract import PLAUSIBLE
from scripts.extract_additional_domain_labs import fresh_unambiguous
from whole_body_jepa.bedside_events import LABELS, nursing_events
from whole_body_jepa.multisource_intervals import sha256


def canonical_bedside(raw, provenance):
    events = nursing_events(raw)
    events = events.merge(provenance[['stay_id', 'uniquepid', 'hospitalid', 'component']],
                          on='stay_id', how='left', validate='many_to_one')
    if events[['uniquepid', 'hospitalid', 'component']].isna().any().any():
        raise ValueError('Nursing event has no candidate provenance')
    events['quality_valid'] = False
    for variable, group in events.groupby('variable'):
        events.loc[group.index, 'quality_valid'] = group.value.between(*PLAUSIBLE[variable])
    events['event_time'] = events.available_at
    return events


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    inventory = Path('artifacts/native_reconstruction_additional_development_domains_20260921.json')
    output = Path('artifacts/native_reconstruction_additional_domain_bedside_20260921.json')
    if output.exists():
        raise ValueError('Completed extraction already exists')
    report = json.loads(inventory.read_text())
    for path, digest in report['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed inventory dependency')
    pp = root/'provenance.parquet'
    if sha256(pp) != report['private_sha256']:
        raise ValueError('Changed candidate provenance')
    patients = pd.read_parquet(pp)
    patients['stay_id'] = 'eicu:' + patients.patientunitstayid
    if patients.stay_id.duplicated().any():
        raise ValueError('Duplicate stay provenance')
    source = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/nurseCharting.csv.gz')
    paths = [Path(__file__), inventory, pp, source,
             Path('scripts/extract_additional_domain_labs.py'),
             Path('whole_body_jepa/bedside_events.py'),
             Path('whole_body_jepa/raw_lab_replay.py'),
             Path('extractors/eicu_sepsis_transition_extract.py')]
    bindings = {str(path): sha256(path) for path in paths}
    folder = root/'bedside'; folder.mkdir(exist_ok=True)
    raw_path = folder/'raw.parquet'; metadata = folder/'raw.json'
    if raw_path.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved['bindings'] != bindings or saved['sha256'] != sha256(raw_path):
            raise ValueError('Changed raw nursing cache')
    else:
        labels = pd.DataFrame(LABELS, columns=['label', 'name', 'variable', 'unit'])
        db = duckdb.connect()
        db.execute('SET threads=2'); db.execute("SET memory_limit='2GB'")
        db.register('patients', patients); db.register('labels', labels)
        db.read_csv(str(source), header=True, all_varchar=True).create_view('nursing')
        temporary = raw_path.with_suffix('.tmp')
        db.sql("""SELECT n.nursingchartid,n.patientunitstayid,
            n.nursingchartoffset,n.nursingchartentryoffset,
            n.nursingchartcelltypecat,n.nursingchartcelltypevallabel,
            n.nursingchartcelltypevalname,n.nursingchartvalue,p.hospitalid
            FROM nursing n JOIN patients p USING(patientunitstayid)
            JOIN labels m ON n.nursingchartcelltypevallabel=m.label
                AND n.nursingchartcelltypevalname=m.name
            WHERE n.nursingchartcelltypecat='Vital Signs'
                AND try_cast(n.nursingchartvalue AS DOUBLE) IS NOT NULL
            """).write_parquet(str(temporary))
        db.close(); temporary.replace(raw_path)
        metadata.write_text(json.dumps(dict(bindings=bindings, sha256=sha256(raw_path)), indent=2)+'\n')
    shards = []; support = []
    total_raw = total_canonical = total_valid = total_fresh = 0
    for number, site in enumerate(sorted(patients.hospitalid.unique())):
        raw = pd.read_parquet(raw_path, filters=[('hospitalid', '==', site)])
        events = canonical_bedside(raw, patients)
        # Select the latest draw BEFORE removing invalid values: no stale rescue.
        eligible = fresh_unambiguous(events)
        eligible = eligible.loc[eligible.quality_valid]
        total_raw += len(raw); total_canonical += len(events)
        total_valid += int(events.quality_valid.sum()); total_fresh += len(eligible)
        support.append(eligible[['uniquepid', 'component', 'variable']].drop_duplicates())
        destination = folder/f'events_{number:03d}.parquet'
        temporary = destination.with_suffix('.tmp')
        events.to_parquet(temporary, index=False); temporary.replace(destination)
        shards.append(dict(path=str(destination), hospitalid=site,
                           sha256=sha256(destination), events=len(events)))
    counts = pd.concat(support, ignore_index=True).drop_duplicates()
    ordinal = {c: i for i, c in enumerate(sorted(patients.component.unique()))}
    targets = {}
    for variable in sorted({row[2] for row in LABELS}):
        part = counts.loc[counts.variable.eq(variable)]
        domains = part.groupby('component').uniquepid.nunique()
        targets[variable] = dict(patients=int(part.uniquepid.nunique()),
            components=len(domains), components_at_least30=int((domains>=30).sum()),
            component_support=[dict(component_ordinal=ordinal[c], patients=int(n))
                               for c, n in domains.items()])
    ip = folder/'event_index.pt'; torch.save(shards, ip)
    sp = folder/'patient_target_support.parquet'; counts.to_parquet(sp, index=False)
    summary = dict(candidate_patients=int(patients.uniquepid.nunique()),
        raw_numeric_mapped_events=total_raw, canonical_time_valid_events=total_canonical,
        canonical_quality_valid_events=total_valid, fresh_unambiguous_event_rows=total_fresh,
        patients_with_fresh_bedside=int(counts.uniquepid.nunique()),
        variables_with_fresh_bedside=sum(v['patients']>0 for v in targets.values()),
        variables_with_six_supported_components=sum(v['components_at_least30']>=6 for v in targets.values()))
    output.write_text(json.dumps(dict(summary=summary, targets=targets, bindings=bindings,
        private_index_sha256=sha256(ip), private_support_sha256=sha256(sp),
        runtime_modified=False, promotion_allowed=False,
        limitation='Unassigned source-domain support only, not independent masked histories, selected-cohort eligibility, model efficacy or a new untouched lockbox. Existing label-convention units, entry>=measurement, quality bounds and <=1h label freshness retained.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

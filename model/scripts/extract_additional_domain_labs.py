"""Canonical local lab extraction for unassigned development-domain candidates."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import torch

from extractors.eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from extractors.mimiciv_observation_transition_extract import EXTRA_PLAUSIBLE
from whole_body_jepa.availability_inputs import UNITS, filter_units
from whole_body_jepa.multisource_intervals import sha256


def fresh_unambiguous(events):
    """Same availability/variable group: newest draw, no conflicting values."""
    keys = ['stay_id', 'available_at', 'variable']
    latest = events.groupby(keys).measured_at.transform('max')
    selected = events.loc[events.measured_at.eq(latest)].copy()
    conflict = selected.groupby(keys).value.transform('nunique').gt(1)
    age = (selected.available_at-selected.measured_at).dt.total_seconds()/3600
    return selected.loc[~conflict & age.between(0,1) & np.isfinite(selected.value)]


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    inventory = Path('artifacts/native_reconstruction_additional_development_domains_20260921.json')
    output = Path('artifacts/native_reconstruction_additional_domain_labs_20260921.json')
    if output.exists():
        raise ValueError('Completed extraction already exists')
    report = json.loads(inventory.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed inventory dependency')
    pp = root/'provenance.parquet'
    if sha256(pp) != report['private_sha256']:
        raise ValueError('Changed candidate provenance')
    patients = pd.read_parquet(pp)
    patients['stay_id'] = 'eicu:'+patients.patientunitstayid
    if patients.patientunitstayid.duplicated().any():
        raise ValueError('Duplicate stay provenance')
    cp = Path('models/whole_body_joint_jepa.pt')
    variables = torch.load(cp, weights_only=False)['variables']
    source = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/lab.csv.gz')
    paths = [Path(__file__), inventory, pp, cp, source, Path('whole_body_jepa/availability_inputs.py'),
             Path('whole_body_jepa/raw_lab_replay.py'),Path('extractors/eicu_sepsis_transition_extract.py'),
             Path('extractors/mimiciv_observation_transition_extract.py')]
    bindings = {str(p):sha256(p) for p in paths}
    folder = root/'labs'; folder.mkdir(exist_ok=True)
    meta = folder/'raw.json'; raw = folder/'raw.parquet'
    if raw.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved['bindings'] != bindings or saved['sha256'] != sha256(raw):
            raise ValueError('Changed raw lab cache')
    else:
        mapping = pd.DataFrame([(k,v) for k,v in LAB_TO_STATE.items() if v in variables and v in UNITS], columns=['name','variable'])
        db = duckdb.connect(); db.execute('SET threads=2'); db.execute("SET memory_limit='2GB'")
        db.register('patients',patients); db.register('mapping',mapping)
        db.read_csv(str(source),header=True).create_view('source')
        tmp = raw.with_suffix('.tmp')
        db.sql("""SELECT p.stay_id,p.uniquepid,p.hospitalid,p.component,
            cast(l.labid as varchar) source_id,m.variable,l.labname source_test_name,
            timestamp '2100-01-01'+l.labresultoffset*interval '1 minute' measured_at,
            timestamp '2100-01-01'+greatest(l.labresultoffset,l.labresultrevisedoffset)*interval '1 minute' available_at,
            try_cast(l.labresult as double) AS value,
            coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface) unit
            FROM source l JOIN patients p ON cast(l.patientunitstayid as varchar)=p.patientunitstayid
            JOIN mapping m ON l.labname=m.name WHERE l.labresultrevisedoffset IS NOT NULL""").write_parquet(str(tmp))
        db.close(); tmp.replace(raw)
        meta.write_text(json.dumps(dict(bindings=bindings,sha256=sha256(raw)),indent=2)+'\n')
    bounds = {**PLAUSIBLE, **EXTRA_PLAUSIBLE}
    shards = []; support = []; total_raw = 0; total_valid = 0; total_fresh = 0
    for number,site in enumerate(sorted(patients.hospitalid.unique())):
        destination = folder/f'events_{number:03d}.parquet'
        frame = pd.read_parquet(raw,filters=[('hospitalid','==',site)])
        total_raw += len(frame)
        clean = filter_units(frame,bounds)
        clean['event_time'] = clean.available_at; clean['quality_valid'] = True
        clean['source_kind'] = 'lab_available'
        eligible = fresh_unambiguous(clean)
        total_valid += len(clean); total_fresh += len(eligible)
        support.append(eligible[['uniquepid','component','variable']].drop_duplicates())
        tmp = destination.with_suffix('.tmp'); clean.to_parquet(tmp,index=False); tmp.replace(destination)
        shards.append(dict(path=str(destination),hospitalid=site,sha256=sha256(destination),events=len(clean)))
    counts = pd.concat(support,ignore_index=True).drop_duplicates()
    ordinal = {c:i for i,c in enumerate(sorted(patients.component.unique()))}
    targets = {}
    for target in variables:
        part = counts.loc[counts.variable.eq(target)]
        domain_counts = part.groupby('component').uniquepid.nunique()
        targets[target] = dict(patients=int(part.uniquepid.nunique()),components=len(domain_counts),
            components_at_least30=int((domain_counts>=30).sum()),
            component_support=[dict(component_ordinal=ordinal[c],patients=int(n)) for c,n in domain_counts.items()])
    ip = folder/'event_index.pt'; torch.save(shards,ip)
    support_path = folder/'patient_target_support.parquet'; counts.to_parquet(support_path,index=False)
    summary = dict(candidate_patients=int(patients.uniquepid.nunique()),raw_mapped_events=total_raw,
        canonical_unit_quality_events=total_valid,fresh_unambiguous_event_rows=total_fresh,
        patients_with_fresh_lab=int(counts.uniquepid.nunique()),
        variables_with_fresh_lab=sum(v['patients']>0 for v in targets.values()),
        variables_with_six_supported_components=sum(v['components_at_least30']>=6 for v in targets.values()))
    output.write_text(json.dumps(dict(summary=summary,targets=targets,bindings=bindings,
        private_index_sha256=sha256(ip),private_support_sha256=sha256(support_path),
        runtime_modified=False,promotion_allowed=False,
        limitation='Lab support/extraction only. Candidate domains remain unassigned. Fresh labels alone do not establish independent target-masked history, full bedside coverage, selected-cohort eligibility or model efficacy. Same existing unit whitelist, plausible bounds, explicit revised-offset availability and <=1h label-age rule.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()

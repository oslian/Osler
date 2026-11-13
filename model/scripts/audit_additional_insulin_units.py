"""Separate raw documentation coverage from accepted drug-dose rate support."""
import json
from pathlib import Path
import duckdb
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    pp = Path('data/research_recovery/additional_development_domains_20260921/provenance.parquet')
    parent = Path('artifacts/native_reconstruction_additional_insulin_support_20260921.json')
    output = Path('artifacts/native_reconstruction_additional_insulin_units_20260921.json')
    source = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    if output.exists():
        raise ValueError('Completed audit already exists')
    prior = json.loads(parent.read_text())
    for path in (pp, source):
        if sha256(path) != prior['bindings'][str(path)]:
            raise ValueError('Changed cohort or raw source')
    patients = pd.read_parquet(pp)
    db = duckdb.connect(); db.execute('SET threads=2')
    db.register('patients', patients)
    db.read_csv(str(source), header=True, all_varchar=True).create_view('rates')
    groups = db.sql("""SELECT r.drugname AS drug_name,count(*) AS event_count,
        count(distinct p.uniquepid) patients,count(distinct p.hospitalid) hospitals,
        count(distinct p.component) components
        FROM rates r JOIN patients p USING(patientunitstayid)
        WHERE lower(r.drugname) LIKE '%insulin%'
        GROUP BY r.drugname ORDER BY event_count DESC""").fetchdf()
    total = db.sql("""SELECT count(*) AS event_count,count(distinct p.uniquepid) patients,
        count(distinct p.component) components
        FROM rates r JOIN patients p USING(patientunitstayid)
        WHERE lower(r.drugname) LIKE '%insulin%'""").fetchdf().iloc[0].to_dict()
    db.close()
    output.write_text(json.dumps(dict(raw_insulin_documentation={k:int(v) for k,v in total.items()},
        label_groups=groups.to_dict('records'),
        accepted_rate_support=prior['summary'],
        bindings={str(p):sha256(p) for p in (Path(__file__),pp,parent,source)},
        runtime_modified=False,promotion_allowed=False,
        limitation='Raw label counts are not validated dose/availability evidence. ml/hr is solution flow, not units/hr without verified concentration and unit semantics. Patients may occur in multiple label groups. No dose conversion, disease diagnosis or inference-time subtype assignment.'),indent=2)+'\n')
    print(json.dumps(dict(total=total,labels=groups.to_dict('records')),indent=2),flush=True)


if __name__=='__main__':
    main()

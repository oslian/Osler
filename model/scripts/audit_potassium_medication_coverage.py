"""Source-only lexical and order support audit; never infer administration."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import classify_action
from scripts.audit_potassium_treatment_support import window_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    if output.exists(): raise ValueError('fresh output required')
    label = Path('/private/tmp/auxiliary_supervision_20260910.npz')
    assets = load_release(Path('models')); col = assets.checkpoint['variables'].index('potassium')
    y = np.load(label); valid = y['paired'][:,col]
    q = pd.DataFrame(dict(stay_id=pd.Series(y['stay_id'][valid]).str.replace('eicu:','',regex=False), offset=(pd.to_datetime(y['query_time'][valid])-pd.Timestamp('2100-01-01')).total_seconds()/60))
    ids = set(q.stay_id.astype(int)); config = get_body_system_config('electrolyte_acid_base')
    root = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    bindings = [label, Path(__file__), Path('extractors/eicu_body_system_configs.py'), Path('extractors/eicu_body_system_transition_extract.py')]
    report = dict(complete=False, status='order_coverage_not_administration', runtime_modified=False, external_data_read=False, queries=len(q), sources={}, binding={str(p):sha256(p) for p in bindings})
    for filename, timecol, idcol in [('infusionDrug.csv.gz','infusionoffset','infusiondrugid'), ('medication.csv.gz','drugorderoffset','medicationid')]:
        path = root/filename; records=[]; total=0
        for chunk in pd.read_csv(path, chunksize=500000, low_memory=False):
            chunk = chunk[chunk.patientunitstayid.isin(ids)]; total += len(chunk)
            # Broad lexical screen, including possible brand aliases for inspection.
            text = chunk.drugname.fillna('').astype(str)
            chunk = chunk[text.str.contains(r'potass|kcl|k-dur|klor|k-lyte', case=False, regex=True)]
            for row in chunk.to_dict('records'):
                name=str(row['drugname']); offset=pd.to_numeric(row[timecol],errors='coerce')
                records.append(dict(stay_id=str(row['patientunitstayid']), source_id=str(row[idcol]), offset=offset,
                    label=name, classified='potassium_repletion' in classify_action(config,name),
                    canceled=str(row.get('drugordercancelled','')).strip().lower()=='yes'))
        events=pd.DataFrame(records).drop_duplicates('source_id')
        eligible=events.loc[~events.canceled & np.isfinite(events.offset)]
        classified=eligible.loc[eligible.classified]
        report['sources'][filename]=dict(cohort_rows=total, lexical_rows=len(events), canceled_rows=int(events.canceled.sum()),
            eligible_rows=len(eligible), classified_rows=len(classified),
            six_hour_windows=int(window_support(q,eligible).sum()), classified_six_hour_windows=int(window_support(q,classified).sum()),
            unmatched_label_counts=events.loc[~events.classified,'label'].value_counts().astype(int).to_dict(),
            semantics='retrospective_infusion_event' if filename.startswith('infusion') else 'order_not_administration',
            independently_verified_availability=False)
        report['binding'][str(path)] = sha256(path)
    report['complete']=True
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['sources'],indent=2))


if __name__=='__main__':main()

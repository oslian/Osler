"""Build private recorded-time pressor snapshots with explicit online abstention."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_map_pressor_inventory import TERMS
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.recorded_pressor_context import recorded_snapshots, DRUGS, DIMENSIONS
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    output = Path('artifacts/map_recorded_pressors_20260912.json')
    cache = root / 'map_recorded_pressors.npz'
    if output.exists() or cache.exists(): raise ValueError('fresh output required')
    q = pd.read_parquet(root / 'queries.parquet')
    anchors = pd.DataFrame({'stay': q.stay_id.str.removeprefix('eicu:').astype(int), 'offset': q.offset})
    stays = set(anchors.stay); records = []
    for chunk in pd.read_csv(raw, chunksize=500000, low_memory=False):
        chunk = chunk[chunk.patientunitstayid.isin(stays)]
        names = chunk.drugname.fillna('').str.lower()
        for drug, pattern in TERMS.items():
            for row in chunk[names.str.contains(pattern, regex=True)].itertuples(index=False):
                offset = pd.to_numeric(row.infusionoffset, errors='coerce')
                if not np.isfinite(offset): continue
                value = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate',
                    rate_like=row.drugrate, original_label=row.drugname), 'vasopressor')
                records.append((int(row.patientunitstayid), row.infusiondrugid, drug,
                    value['action_rate_dimension'], float(offset), value['action_rate']))
    events = pd.DataFrame(records, columns=['stay', 'id', 'drug', 'dimension', 'offset', 'rate'])
    result = recorded_snapshots(anchors, events)
    np.savez_compressed(cache, **result, drugs=np.array(DRUGS), dimensions=np.array(DIMENSIONS))
    eligible = np.load(root / 'map_all_query_labels.npz')['paired'][:, 0]
    summary = {}
    for j, drug in enumerate(DRUGS):
        status, counts = np.unique(result['status'][eligible, j], return_counts=True)
        summary[drug] = dict(zip(status.tolist(), counts.tolist()))
    paths = [Path(__file__), Path('whole_body_jepa/recorded_pressor_context.py'),
        Path('whole_body_jepa/event_treatment.py'), Path('osler_jepa/treatment_units.py'),
        Path('scripts/audit_map_pressor_inventory.py'), raw, root/'queries.parquet',
        root/'map_all_query_labels.npz', cache]
    output.write_text(json.dumps(dict(status='retrospective_features_only', queries=len(q),
        event_rows=len(events), map_eligible_queries=int(eligible.sum()), drug_status=summary,
        online_eligible=False, training_performed=False, runtime_modified=False,
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    print(output.read_text())


if __name__ == '__main__': main()

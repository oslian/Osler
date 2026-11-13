"""Recorded-time pressor history at fixed past cutoffs; not delivered-dose history."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_map_pressor_inventory import TERMS
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.recorded_pressor_context import recorded_snapshots, DRUGS, DIMENSIONS
from whole_body_jepa.multisource_intervals import sha256


def documented_change(current, previous, lag_hours):
    gap = lag_hours + previous['age_hours'] - current['age_hours']
    valid = current['known'] & previous['known'] & (gap[..., None] > 0)
    delta = np.where(valid, current['rate'] - previous['rate'], np.nan)
    return delta, valid, gap


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    output = Path('artifacts/map_pressor_history_20260912.json'); cache = root / 'map_pressor_history.npz'
    if output.exists() or cache.exists(): raise ValueError('fresh output required')
    parent = Path('artifacts/map_recorded_pressors_20260912.json')
    for p, h in json.loads(parent.read_text())['bindings'].items(): assert sha256(Path(p)) == h
    q = pd.read_parquet(root / 'queries.parquet')
    anchors = pd.DataFrame({'stay':q.stay_id.str.removeprefix('eicu:').astype(int), 'offset':q.offset})
    records = []; stays = set(anchors.stay)
    for c in pd.read_csv(raw, chunksize=500000, low_memory=False):
        c = c[c.patientunitstayid.isin(stays)]; names = c.drugname.fillna('').str.lower()
        for drug, pattern in TERMS.items():
            for r in c[names.str.contains(pattern, regex=True)].itertuples(index=False):
                offset = pd.to_numeric(r.infusionoffset, errors='coerce')
                if not np.isfinite(offset): continue
                rate = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate',
                    rate_like=r.drugrate, original_label=r.drugname), 'vasopressor')
                records.append((int(r.patientunitstayid), r.infusiondrugid, drug,
                    rate['action_rate_dimension'], float(offset), rate['action_rate']))
    events = pd.DataFrame(records, columns=['stay','id','drug','dimension','offset','rate'])
    current = recorded_snapshots(anchors, events)
    saved = np.load(root/'map_recorded_pressors.npz')
    for key in ('rate','known','age_hours','status'): np.testing.assert_array_equal(current[key], saved[key])
    payload = {}; results = {}; eligible = np.load(root/'map_all_query_labels.npz')['paired'][:,0]
    for lag in (1, 3):
        shifted = anchors.copy(); shifted['offset'] -= lag * 60
        previous = recorded_snapshots(shifted, events)
        delta, valid, gap = documented_change(current, previous, lag)
        for key in ('rate','known','age_hours','status'): payload[f'{key}_lag{lag}'] = previous[key]
        payload[f'delta_lag{lag}'] = delta; payload[f'delta_known_lag{lag}'] = valid
        payload[f'record_gap_hours_lag{lag}'] = gap
        results[str(lag)] = {}
        for j, drug in enumerate(DRUGS):
            pair = valid[:,j].any(axis=1) & eligible
            changed = (np.isfinite(delta[:,j]) & (np.abs(delta[:,j]) > 1e-8)).any(axis=1) & eligible
            results[str(lag)][drug] = dict(paired_queries=int(pair.sum()), changed_queries=int(changed.sum()),
                changed_patients=int(q.loc[changed,'uniquepid'].nunique()))
    np.savez_compressed(cache, **payload)
    output.write_text(json.dumps(dict(status='retrospective_history_only', current_reproduced=True,
        results=results, online_eligible=False, runtime_modified=False, training_performed=False,
        limitations=['Changes are differences between recorded snapshots, not confirmed delivered-rate changes.',
                     'Different dimensions or identical underlying record times do not yield a delta.',
                     'Independent availability remains unverified.'],
        bindings={str(p):sha256(p) for p in [Path(__file__), parent, raw, root/'queries.parquet',
            root/'map_all_query_labels.npz', cache]}),indent=2)+'\n')
    print(output.read_text())


if __name__ == '__main__': main()

"""MAP availability-native support, preserving original patient partitions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.raw_lab_replay import canonical_events


def summary(q, labels):
    valid = labels['paired'][:, 0]
    def quantiles(values):
        return np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist() if len(values) else []
    ids = set(zip(q.loc[valid, 'stay_id'], labels['future_source_id'][valid, 0]))
    return dict(queries=len(q), pairs=int(valid.sum()),
                patients=int(q.loc[valid, 'uniquepid'].nunique()),
                unique_future_events=len(ids),
                age_hours=quantiles(labels['measurement_age'][valid, 0]),
                actual_horizon_hours=quantiles(labels['actual_delta_t'][valid, 0]))


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    private = root / 'map_native_anchor_support'
    output = Path('artifacts/map_native_anchor_support_20260912.json')
    if output.exists() or private.exists():
        raise ValueError('fresh outputs required')
    oldq = pd.read_parquet(root / 'queries.parquet')
    old = dict(np.load(root / 'map_all_query_labels.npz'))
    mapping = oldq[['stay_id', 'uniquepid', 'hospitalid']].drop_duplicates()
    if mapping.stay_id.duplicated().any():
        raise ValueError('ambiguous stay provenance')
    events = canonical_events(pd.read_parquet(root / 'qualified_bedside_events.parquet'))
    events = events.loc[events.variable.eq('map') & events.stay_id.isin(mapping.stay_id)]
    # Keep every arrival, including invalid arrivals: label eligibility handles
    # quality/conflicts without conditioning query generation on future truth.
    q = events[['stay_id', 'available_at']].rename(columns={'available_at': 't'}).drop_duplicates()
    q = q.merge(mapping, on='stay_id', validate='many_to_one')
    q = q.sort_values(['stay_id', 't']).reset_index(drop=True)
    q['offset'] = (q.t - pd.Timestamp('2100-01-01')).dt.total_seconds() / 60
    labels = future_pairs(q, events, ['map'], horizon=6, tolerance=1, max_age=6)
    # Independently rebuild the old labels under the identical event contract.
    replay = future_pairs(oldq, events, ['map'], horizon=6, tolerance=1, max_age=6)
    for name in replay:
        np.testing.assert_array_equal(replay[name], old[name])
    old_valid, valid = old['paired'][:, 0], labels['paired'][:, 0]
    old_ids = set(zip(oldq.loc[old_valid, 'stay_id'], old['future_source_id'][old_valid, 0]))
    new_ids = set(zip(q.loc[valid, 'stay_id'], labels['future_source_id'][valid, 0]))
    old_keys = set(zip(oldq.stay_id, oldq.t))
    new_keys = set(zip(q.stay_id, q.t))
    private.mkdir()
    bindings = [Path(__file__), root/'queries.parquet', root/'map_all_query_labels.npz',
                root/'qualified_bedside_events.parquet',
                Path('scripts/build_rate_anchored_potassium.py'),
                Path('whole_body_jepa/availability_supervision.py'),
                Path('whole_body_jepa/availability_inputs.py'),
                Path('whole_body_jepa/raw_lab_replay.py')]
    counts = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        path = root / f'partition_{seed}.npz'
        roles = dict(np.load(path))
        patient_sets = {k: set(oldq.loc[m, 'uniquepid']) for k, m in roles.items()}
        sets = list(patient_sets.values())
        assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i+1:])
        new_roles = {k: q.uniquepid.isin(patients).to_numpy() for k, patients in patient_sets.items()}
        np.savez_compressed(private/f'partition_{seed}.npz', **new_roles)
        counts.append(dict(seed=seed, roles={k: dict(rows=int((m & valid).sum()),
            patients=int(q.loc[m & valid, 'uniquepid'].nunique())) for k, m in new_roles.items()}))
        bindings.extend([path, private/f'partition_{seed}.npz'])
    q.to_parquet(private/'queries.parquet', index=False)
    np.savez_compressed(private/'labels.npz', **labels)
    bindings.extend([private/'queries.parquet', private/'labels.npz'])
    report = dict(complete=True, old=summary(oldq, old), native=summary(q, labels),
                  identical_query_times=len(old_keys & new_keys), common_future_events=len(old_ids & new_ids),
                  quantile_probabilities=[0, .25, .5, .75, .95, 1], old_labels_exact=True,
                  roles=counts, training_performed=False, runtime_modified=False, promotion_allowed=False,
                  limitation='Same restricted stay universe, not all eICU. Changed task and weighting; not paired accuracy repair.',
                  bindings={str(p): sha256(p) for p in bindings})
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('roles','bindings')}, indent=2))


if __name__ == '__main__':
    main()

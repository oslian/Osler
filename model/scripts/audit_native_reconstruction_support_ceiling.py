"""Input-only patient support ceiling; not reconstruction eligibility or efficacy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    cache = root / 'map_native_anchor_support'
    output = Path('artifacts/native_reconstruction_support_ceiling_20260913.json')
    if output.exists():
        raise ValueError('existing evidence must not be overwritten')
    q = pd.read_parquet(cache / 'queries.parquet')
    partition = dict(np.load(cache / 'partition_7.npz'))
    roles = {role: set(q.loc[partition[role], 'uniquepid'])
             for role in ('fit', 'tune', 'evaluation', 'hospital')}
    if sum(map(len, roles.values())) != len(set.union(*roles.values())):
        raise ValueError('overlapping patient roles')
    link = q[['stay_id', 'uniquepid']].drop_duplicates()
    if link.stay_id.duplicated().any():
        raise ValueError('ambiguous stay identity')
    labs = pd.read_parquet(root / 'native_events.parquet')
    labs['quality_valid'] = True
    bedside = pd.read_parquet(root / 'qualified_bedside_events.parquet')
    bedside['event_time'] = bedside.available_at
    events = pd.concat([labs, bedside], ignore_index=True)
    events = events.loc[events.source_kind.isin(['lab_available', 'bedside_available'])]
    events = events.merge(link, on='stay_id', how='inner', validate='many_to_one')
    lag = (pd.to_datetime(events.event_time) - pd.to_datetime(events.measured_at)).dt.total_seconds() / 3600
    valid = (lag.between(0, 1) & events.quality_valid.fillna(False).astype(bool)
             & np.isfinite(pd.to_numeric(events.value, errors='coerce')))
    variables = load_release(Path('models')).checkpoint['variables']
    counts = {}
    for role, people in roles.items():
        supported = events.loc[valid & events.uniquepid.isin(people)]
        counts[role] = {v: int(supported.loc[supported.variable.eq(v), 'uniquepid'].nunique())
                        for v in variables}
    paths = [Path(__file__), root / 'native_events.parquet', root / 'qualified_bedside_events.parquet',
             cache / 'queries.parquet', cache / 'partition_7.npz']
    report = dict(complete=True, role_patients={r: len(p) for r, p in roles.items()},
                  target_patient_ceiling=counts,
                  below_minimum_in_any_role=[v for v in variables if any(counts[r][v] < 30 for r in roles)],
                  eligibility_verified=False, runtime_modified=False, promotion_allowed=False,
                  limitation='Upper bound only: no source-version conflict resolution or independent masked history check. MAP-derived cohort, not all eICU.',
                  bindings={str(p): sha256(p) for p in paths})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

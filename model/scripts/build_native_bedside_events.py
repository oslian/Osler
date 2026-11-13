"""Build private nursing events and report unchanged-query observation support."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from whole_body_jepa.availability_inputs import snapshots
from whole_body_jepa.bedside_events import LABELS, nursing_events
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    source = root / 'numeric_nurse_charting.parquet'
    destination = root / 'canonical_bedside_events.parquet'
    output = Path('artifacts/native_bedside_events_20260911.json')
    if destination.exists() or output.exists():
        raise ValueError('fresh outputs required')
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.read_parquet(str(source)).create_view('nursing')
    raw = con.sql("SELECT * FROM nursing WHERE nursingchartcelltypecat = 'Vital Signs'").df()
    events = nursing_events(raw)
    events.to_parquet(destination, index=False)
    q = pd.read_parquet(root / 'queries.parquet').loc[np.load(root / 'labels.npz')['paired'][:, 0]].reset_index(drop=True)
    variables = list(dict.fromkeys(item[2] for item in LABELS))
    observations = np.full((len(q), len(variables)), np.nan)
    event_groups = dict(tuple(events.groupby('stay_id', sort=False)))
    for stay, group in q.groupby('stay_id', sort=False):
        if stay in event_groups:
            ordered = group.sort_values('t')
            values, _ = snapshots(event_groups[stay], ordered.t, variables, 6.)
            observations[ordered.index] = values
    report = dict(complete=True, runtime_modified=False, training_performed=False,
        raw_vital_rows=len(raw), canonical_rows=len(events), queries=len(q),
        freshness_hours=6, scope='timestamp-qualified observations, not range-validated model inputs',
        unit_policy='C explicit; other units label conventions, no explicit raw unit column',
        variables={v: dict(events=int(events.variable.eq(v).sum()),
            supported_queries=int(np.isfinite(observations[:, i]).sum()),
            supported_patients=int(q.loc[np.isfinite(observations[:, i]), 'uniquepid'].nunique()),
            zero_events=int((events.variable.eq(v) & events.value.eq(0)).sum()))
            for i, v in enumerate(variables)},
        bindings={str(p): sha256(p) for p in [source, destination, root/'queries.parquet',
            root/'labels.npz', Path(__file__), Path('whole_body_jepa/bedside_events.py')]})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'bindings'}, indent=2))


if __name__ == '__main__':
    main()

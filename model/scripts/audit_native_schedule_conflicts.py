"""Trace known conflicting arrivals through the actual cached token schedule."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.raw_lab_replay import canonical_events


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_schedule_conflicts_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    paths = [root / name for name in ('native_events.parquet', 'native_schedule.npz',
             'native_tokens.npz', 'body_labs.parquet', 'queries.parquet', 'labels.npz')]
    reference = json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
    for path in paths[:3]:
        if sha256(path) != reference['binding'][str(path)]:
            raise ValueError('original native cache changed')
    raw = canonical_events(pd.read_parquet(paths[3]))
    events = pd.read_parquet(paths[0])
    schedule = dict(np.load(paths[1]))
    tokens = dict(np.load(paths[2]))
    variables = np.load(root / 'multitarget_labels.npz')['variables'].tolist()
    n = len(variables)
    if tokens['input'].shape[1] < 3 * n:
        raise ValueError('token schema mismatch')
    np.testing.assert_array_equal(schedule['query_ptr'], tokens['query_ptr'])
    if len(tokens['history_indices']) != len(schedule['group_ptr']) - 1:
        raise ValueError('token schedule alignment mismatch')
    queries = pd.read_parquet(paths[4]).loc[np.load(paths[5])['paired'][:, 0]].reset_index(drop=True)

    # Resolve versions only as they arrive, never using a later conflict early.
    keys = ['stay_id', 'variable', 'measured_at']
    counts = raw.groupby(keys).value.nunique()
    selected = raw.set_index(keys).loc[counts[counts > 1].index].reset_index()
    conflict_arrivals = set()
    for key, group in selected.groupby(keys):
        known = {}
        for time, arrival in group.groupby('available_at', sort=True):
            for row in arrival.itertuples():
                known[str(row.source_id)] = float(row.value)
            if len(set(known.values())) > 1:
                conflict_arrivals.add((*key, time))
    risky_ids = set()
    for row in events.itertuples():
        if (row.stay_id, row.variable, row.measured_at, row.event_time) in conflict_arrivals:
            risky_ids.add(row.Index)

    exposed = defaultdict(set)
    exposed_tokens = defaultdict(set)
    checked = {}
    for group_index, (left, right) in enumerate(zip(schedule['group_ptr'][:-1], schedule['group_ptr'][1:])):
        token_id = int(tokens['history_indices'][group_index])
        if token_id not in checked:
            ids = schedule['event_indices'][left:right]
            found = []
            if any(int(i) in risky_ids for i in ids):
                group = events.iloc[ids]
                for variable, part in group.loc[group.source_kind.eq('lab_available')].groupby('variable'):
                    if variable not in variables:
                        continue
                    latest = part.loc[part.measured_at.eq(part.measured_at.max())]
                    if not any(int(i) in risky_ids for i in latest.index):
                        continue
                    if tokens['input'][token_id, n + variables.index(variable)] > 0:
                        found.append(variable)
            checked[token_id] = found
        if checked[token_id]:
            query_index = int(np.searchsorted(schedule['query_ptr'], group_index, side='right') - 1)
            for variable in checked[token_id]:
                exposed[variable].add(query_index)
                exposed_tokens[variable].add(token_id)
    all_queries = set().union(*exposed.values()) if exposed else set()
    report = dict(complete=True, runtime_modified=False, training_performed=False,
        scope='actual unmasked cached tokens with known conflicting sources; not forecast-error attribution',
        queries=len(queries), exposed_queries=len(all_queries),
        exposed_patients=int(queries.iloc[sorted(all_queries)].uniquepid.nunique()),
        variables={v: dict(queries=len(rows), patients=int(queries.iloc[sorted(rows)].uniquepid.nunique()),
                          unique_tokens=len(exposed_tokens[v])) for v, rows in sorted(exposed.items())},
        binding={str(p): sha256(p) for p in paths + [Path(__file__)]})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'binding'}, indent=2))


if __name__ == '__main__':
    main()

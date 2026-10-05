"""Build native-event histories at MAP availability anchors, one stay shard at a time."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.native_history_schedule import conflict_checked_history_schedule
from whole_body_jepa.runtime import load_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    private = root/'map_native_anchor_support'
    source = private/'input_shards'/f'shard_{args.shard:03d}.npz'
    dest = private/'native_query_shards'
    path = dest/source.name
    report_path = Path('artifacts')/f'map_native_query_tokens_{args.shard:03d}_20260912.json'
    if path.exists() or report_path.exists():
        raise ValueError('fresh output required')
    rows = np.load(source)['global_query_rows']
    queries = pd.read_parquet(private/'queries.parquet')
    q = queries.iloc[rows].reset_index(drop=True)
    original = pd.read_parquet(root/'native_events.parquet')
    original['quality_valid'] = True
    bedside = pd.read_parquet(root/'qualified_bedside_events.parquet')
    bedside['event_time'] = bedside.available_at
    events = pd.concat([original, bedside], ignore_index=True)
    events = events.loc[events.stay_id.isin(q.stay_id.unique())]
    events, schedules = conflict_checked_history_schedule(events, q)
    print('schedules ready', len(schedules), flush=True)
    assets = load_release(Path('models'))
    tokens, treatments, ids, ptr, lookup = [], [], [], [0], {}
    for groups in schedules:
        for group in groups:
            key = tuple(group.tolist())
            if key not in lookup:
                token, treatment = encode_group(
                    events.iloc[group], assets.checkpoint['variables'],
                    assets.scaler.medians, assets.scaler.scales,
                    assets.model.variable_module_ids.numpy(),
                    len(assets.checkpoint['modules']), assets.max_age_hours)
                lookup[key] = len(tokens)
                tokens.append(token)
                treatments.append(treatment)
            ids.append(lookup[key])
        ptr.append(len(ids))
    matrix = np.asarray(tokens, np.float32)
    values = np.asarray(treatments, np.float64)
    ids, ptr = np.asarray(ids, np.int64), np.asarray(ptr, np.int64)
    if not np.isfinite(matrix).all():
        raise ValueError('nonfinite input')
    oldq = pd.read_parquet(root/'queries.parquet')
    oldq['old_index'] = np.arange(len(oldq))
    matched = q[['stay_id', 't']].merge(oldq[['stay_id', 't', 'old_index']],
        on=['stay_id', 't'], how='left', validate='one_to_one')
    old = dict(np.load(root/'map_all_native_tokens.npz'))
    compared = 0
    for j, old_index in enumerate(matched.old_index):
        if pd.isna(old_index):
            continue
        i = int(old_index)
        a = ids[ptr[j]:ptr[j+1]]
        b = old['history_indices'][old['query_ptr'][i]:old['query_ptr'][i+1]]
        np.testing.assert_array_equal(matrix[a], old['input'][b])
        np.testing.assert_array_equal(values[a], old['event_values'][b])
        compared += 1
    dest.mkdir(exist_ok=True)
    np.savez_compressed(path, input=matrix, event_values=values,
        history_indices=ids, query_ptr=ptr, global_query_rows=rows)
    bound = [source, private/'queries.parquet', root/'queries.parquet',
        root/'native_events.parquet', root/'qualified_bedside_events.parquet',
        root/'map_all_native_tokens.npz', Path(__file__), path,
        Path('whole_body_jepa/native_history_schedule.py'),
        Path('whole_body_jepa/native_event_tokens.py')]
    report = dict(shard=args.shard, complete=True, queries=len(q),
        common_histories_exact=compared, unique_groups=len(matrix),
        max_history=int(np.diff(ptr).max()),
        median_history=float(np.median(np.diff(ptr))),
        training_performed=False, runtime_modified=False,
        scope='one shard only; insulin record time is not verified administration availability',
        bindings={str(p): sha256(p) for p in bound})
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print({k: v for k, v in report.items() if k != 'bindings'}, flush=True)


if __name__ == '__main__':
    main()

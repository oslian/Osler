"""Build private query-time replay histories; never overwrite old evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_history_schedule import conflict_checked_history_schedule
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_conflict_replay_20260911.json')
    private = root / 'native_schedule_conflict_replay.npz'
    if output.exists() or private.exists():
        raise ValueError('fresh outputs required')
    events = pd.read_parquet(root / 'native_events.parquet')
    queries = pd.read_parquet(root / 'queries.parquet').loc[
        np.load(root / 'labels.npz')['paired'][:, 0]].reset_index(drop=True)
    canonical, schedules = conflict_checked_history_schedule(events, queries)
    pd.testing.assert_frame_equal(events, canonical)
    old = dict(np.load(root / 'native_schedule.npz'))
    query_ptr = [0]; group_ptr = [0]; indices = []; changed = []; removed = 0
    for i, groups in enumerate(schedules):
        ids = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
        start, stop = old['query_ptr'][i:i+2]
        previous = old['event_indices'][old['group_ptr'][start]:old['group_ptr'][stop]]
        if not set(ids).issubset(set(previous)):
            raise ValueError('replay introduced new events')
        if set(ids) != set(previous):
            changed.append(i)
        removed += len(previous) - len(ids)
        for group in groups:
            indices.extend(group.tolist()); group_ptr.append(len(indices))
        query_ptr.append(len(group_ptr) - 1)
    np.savez_compressed(private, query_ptr=np.asarray(query_ptr, np.int64),
                        group_ptr=np.asarray(group_ptr, np.int64), event_indices=np.asarray(indices, np.int64))
    report = dict(complete=True, runtime_modified=False, training_performed=False,
        queries=len(queries), changed_queries=len(changed),
        changed_patients=int(queries.iloc[changed].uniquepid.nunique()), removed_event_occurrences=removed,
        policy='as-of latest source versions; reject disagreeing draws; query-time replay',
        binding={str(p): sha256(p) for p in [private, root/'native_events.parquet', root/'native_schedule.npz',
            root/'queries.parquet', root/'labels.npz', Path(__file__), Path('whole_body_jepa/native_history_schedule.py')]})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'binding'}, indent=2))


if __name__ == '__main__':
    main()

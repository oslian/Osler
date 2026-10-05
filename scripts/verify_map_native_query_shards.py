"""Verify complete MAP-native history coverage and neural input assembly."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_encoder import native_arrays


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output = Path('artifacts/map_native_query_history_complete_20260912.json')
    if output.exists():
        raise ValueError('fresh report required')
    q = pd.read_parquet(root/'queries.parquet')
    labels = np.load(root/'labels.npz')['paired'][:, 0]
    visited = np.zeros(len(q), dtype=np.int64)
    digests, shards = {}, []
    for source in sorted((root/'input_shards').glob('shard_*.npz')):
        index = int(source.stem.split('_')[-1])
        report_path = Path('artifacts')/f'map_native_query_tokens_{index:03d}_20260912.json'
        report = json.loads(report_path.read_text())
        if not report['complete'] or report['shard'] != index:
            raise ValueError('incomplete or wrong shard report')
        for name, expected in report['bindings'].items():
            if name not in digests:
                digests[name] = sha256(Path(name))
            if digests[name] != expected:
                raise ValueError(f'changed dependency: {name}')
        base = dict(np.load(source))
        native = dict(np.load(root/'native_query_shards'/source.name))
        global_rows = native['global_query_rows']
        np.testing.assert_array_equal(global_rows, base['global_query_rows'])
        if len(np.unique(global_rows)) != len(global_rows):
            raise ValueError('duplicate shard queries')
        if (global_rows < 0).any() or (global_rows >= len(q)).any():
            raise ValueError('out of range query')
        visited[global_rows] += 1
        arrays, rows, events = native_arrays(base, native,
            base['current'].shape[1], base['presence'].shape[1])
        if not np.isfinite(arrays['input']).all():
            raise ValueError('nonfinite input')
        for i, row in enumerate(rows):
            assembled = arrays['history'][row]
            assembled = assembled[assembled > 0]
            expected = native['history_indices'][native['query_ptr'][i]:native['query_ptr'][i+1]]
            np.testing.assert_array_equal(arrays['input'][assembled], native['input'][expected])
            np.testing.assert_array_equal(events[assembled], native['event_values'][expected])
        shards.append(dict(shard=index, queries=len(rows),
            eligible_pairs=int(labels[global_rows].sum()),
            common_histories_exact=report['common_histories_exact'],
            max_history=report['max_history'],
            report_sha256=sha256(report_path)))
        print('verified', index, len(rows), flush=True)
    if not len(shards) or not np.all(visited == 1):
        raise ValueError('incomplete or repeated global query coverage')
    result = dict(complete=True, shards=shards, queries=len(q),
        eligible_pairs=int(labels.sum()),
        common_histories_exact=sum(s['common_histories_exact'] for s in shards),
        all_assembled_sequences_exact=True, query_context_exact=True,
        training_performed=False, runtime_modified=False,
        scope='history preparation and assembly only, not accuracy or promotion',
        bindings={**digests, str(Path(__file__)):sha256(Path(__file__)),
            str(root/'labels.npz'):sha256(root/'labels.npz'),
            'whole_body_jepa/native_event_encoder.py':sha256(Path('whole_body_jepa/native_event_encoder.py'))})
    output.write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('shards','bindings')})


if __name__ == '__main__':
    main()

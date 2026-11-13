"""Build MAP-native inputs in bounded stay batches, retaining exact histories."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    private = root/'map_native_anchor_support'
    parent = Path('artifacts/map_native_anchor_support_20260912.json')
    output = Path('artifacts/map_native_anchor_inputs_20260912.json')
    dest = private/'input_shards'
    if dest.exists() or output.exists():
        raise ValueError('fresh outputs required')
    for p, digest in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p)) != digest:
            raise ValueError(f'changed input: {p}')
    q = pd.read_parquet(private/'queries.parquet')
    labels = dict(np.load(private/'labels.npz'))
    oldq = pd.read_parquet(root/'queries.parquet')
    oldq['old_index'] = np.arange(len(oldq))
    matched = q[['stay_id', 't']].merge(oldq[['stay_id', 't', 'old_index']],
                                       on=['stay_id', 't'], how='left', validate='one_to_one')
    old = dict(np.load(root/'map_all_query_inputs.npz'))
    labs = pd.read_parquet(root/'body_labs.parquet')
    labs['quality_valid'] = True
    raw = pd.concat([labs, pd.read_parquet(root/'qualified_bedside_events.parquet')], ignore_index=True)
    assets = load_release(Path('models'))
    col = assets.checkpoint['variables'].index('map')
    stays = q.stay_id.unique()
    dest.mkdir()
    shards = []
    compared = 0
    for start in range(0, len(stays), 32):
        batch_stays = stays[start:start+32]
        global_rows = np.flatnonzero(q.stay_id.isin(batch_stays))
        frame = q.iloc[global_rows].reset_index(drop=True)
        arrays, rows = build_arrays(frame, raw.loc[raw.stay_id.isin(batch_stays)], assets,
                                  {s:pd.Timestamp('2100-01-01') for s in batch_stays})
        valid = labels['paired'][global_rows, 0]
        np.testing.assert_allclose(
            arrays['current'][rows[valid], col]*assets.scaler.scales[col]+assets.scaler.medians[col],
            labels['current'][global_rows[valid], 0], rtol=1e-6, atol=1e-4)
        matches = matched.old_index.iloc[global_rows].to_numpy()
        common = np.isfinite(matches)
        oldrows = old['query_rows'][matches[common].astype(int)]
        for field in ('input', 'current', 'mask', 'ages', 'presence', 'horizon'):
            np.testing.assert_array_equal(arrays[field][rows[common]], old[field][oldrows])
            np.testing.assert_array_equal(arrays[field][arrays['history'][rows[common]]],
                                          old[field][old['history'][oldrows]])
        compared += int(common.sum())
        path = dest/f'shard_{start//32:03d}.npz'
        np.savez_compressed(path, **arrays, query_rows=rows, global_query_rows=global_rows)
        shards.append(dict(path=str(path), sha256=sha256(path), queries=len(frame),
                           snapshots=len(arrays['input']), eligible_pairs=int(valid.sum())))
        print(f'Completed shard {len(shards)}: {len(frame)} queries', flush=True)
    assert sum(x['queries'] for x in shards) == len(q)
    assert compared == int(matched.old_index.notna().sum())
    bindings = [parent, Path(__file__), root/'map_all_query_inputs.npz', root/'queries.parquet',
                root/'body_labs.parquet', root/'qualified_bedside_events.parquet',
                Path('whole_body_jepa/availability_inputs.py')]
    report = dict(complete=True, queries=len(q), eligible_pairs=int(labels['paired'].sum()),
                  common_query_and_history_exact=compared, shards=shards,
                  training_performed=False, runtime_modified=False,
                  scope='fixed-lag input reconstruction, not native event training or promotion',
                  bindings={str(p):sha256(p) for p in bindings})
    output.write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ('shards', 'bindings')})


if __name__ == '__main__':
    main()

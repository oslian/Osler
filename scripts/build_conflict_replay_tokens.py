"""Encode query-as-of replay with the existing native observation projection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_conflict_replay_tokens_20260911.json')
    path = root / 'native_tokens_conflict_replay.npz'
    if output.exists() or path.exists():
        raise ValueError('fresh outputs required')
    reference = json.loads(Path('artifacts/native_conflict_replay_20260911.json').read_text())
    for name, digest in reference['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('replay dependency changed')
    events = pd.read_parquet(root / 'native_events.parquet')
    schedule = dict(np.load(root / 'native_schedule_conflict_replay.npz'))
    assets = load_release(Path('models'))
    tokens = []; treatments = []; indices = []; lookup = {}
    for left, right in zip(schedule['group_ptr'][:-1], schedule['group_ptr'][1:]):
        ids = schedule['event_indices'][left:right]
        key = tuple(ids.tolist())
        if key not in lookup:
            token, treatment = encode_group(events.iloc[ids], assets.checkpoint['variables'],
                assets.scaler.medians, assets.scaler.scales, assets.model.variable_module_ids.numpy(),
                len(assets.checkpoint['modules']), assets.max_age_hours)
            lookup[key] = len(tokens); tokens.append(token); treatments.append(treatment)
        indices.append(lookup[key])
    token = np.asarray(tokens, np.float32)
    if token.shape[1] != assets.checkpoint['input_dim']:
        raise ValueError('input schema mismatch')
    base = dict(np.load(root / 'body_arrays_conflict_checked.npz'))
    labels = dict(np.load(root / 'multitarget_labels_joint_conflict_checked.npz'))
    np.testing.assert_array_equal(labels['variables'], assets.checkpoint['variables'])
    rows = base['query_rows']
    np.testing.assert_array_equal(base['mask'][rows] > 0, np.isfinite(labels['current']))
    known = np.isfinite(labels['current'])
    decoded = base['current'][rows] * assets.scaler.scales + assets.scaler.medians
    np.testing.assert_allclose(decoded[known], labels['current'][known], atol=1e-4, rtol=1e-6)
    if len(rows) != len(schedule['query_ptr']) - 1:
        raise ValueError('query count mismatch')
    np.savez_compressed(path, input=token, event_values=np.asarray(treatments, np.float64),
                        query_ptr=schedule['query_ptr'], history_indices=np.asarray(indices, np.int64))
    report = dict(complete=True, runtime_modified=False, training_performed=False,
        unique_groups=len(token), queries=len(rows), input_width=token.shape[1],
        current_label_mask_exact=True,
        binding={str(p): sha256(p) for p in [path, Path(__file__), root/'native_events.parquet',
            root/'native_schedule_conflict_replay.npz', root/'body_arrays_conflict_checked.npz',
            root/'multitarget_labels_joint_conflict_checked.npz', Path('whole_body_jepa/native_event_tokens.py')]})
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'binding'}, indent=2))


if __name__ == '__main__':
    main()

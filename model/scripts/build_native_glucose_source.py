"""Align source provenance to existing native tokens without changing values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_glucose_source import source_indicator, SOURCE_NAMES
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = root / 'native_glucose_source.npz'
    report_path = Path('artifacts/native_glucose_source_20260911.json')
    if output.exists() or report_path.exists():
        raise ValueError('fresh output required')
    reference = json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
    paths = [root / n for n in ('native_events.parquet', 'native_schedule.npz', 'native_tokens.npz')]
    for p in paths:
        if sha256(p) != reference['binding'][str(p)]:
            raise ValueError('native input changed')
    metadata_path = root / 'glucose_source_metadata.parquet'
    metadata = pd.read_parquet(metadata_path)[['source_id', 'labname']]
    metadata['source_id'] = metadata.source_id.astype(str)
    if metadata.source_id.duplicated().any():
        raise ValueError('source ID not unique')
    events = pd.read_parquet(paths[0])
    events['source_test_name'] = events.source_id.astype(str).map(metadata.set_index('source_id').labname)
    # IDs are local to their table: never assign lab provenance to rate events.
    events.loc[~events.source_kind.eq('lab_available'), 'source_test_name'] = None
    with np.load(paths[1]) as f:
        schedule = {k: f[k] for k in f.files}
    with np.load(paths[2]) as f:
        native = {k: f[k] for k in f.files}
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    col = variables.index('glucose')
    observed = native['input'][:, len(variables) + col].astype(bool)
    lookup, indices, features = {}, [], []
    for left, right in zip(schedule['group_ptr'][:-1], schedule['group_ptr'][1:]):
        ids = schedule['event_indices'][left:right]
        key = tuple(ids.tolist())
        if key not in lookup:
            index = len(features)
            lookup[key] = index
            features.append(source_indicator(events.iloc[ids], observed[index]))
        indices.append(lookup[key])
    np.testing.assert_array_equal(indices, native['history_indices'])
    np.testing.assert_array_equal(schedule['query_ptr'], native['query_ptr'])
    features = np.asarray(features, np.float32)
    assert features.shape == (len(native['input']), 2)
    assert not features[~observed].any()
    np.savez_compressed(output, source_indicators=features,
                        history_indices=native['history_indices'], query_ptr=native['query_ptr'])
    report = dict(complete=True, training_performed=False, runtime_modified=False,
                  causal_claim_allowed=False, columns=list(SOURCE_NAMES), groups=len(features),
                  observed_glucose_groups=int(observed.sum()),
                  source_counts=features.sum(0).astype(int).tolist(),
                  observed_unknown_groups=int((observed & ~features.any(1)).sum()),
                  binding={str(p): sha256(p) for p in paths + [metadata_path, output, Path(__file__),
                           Path('whole_body_jepa/native_glucose_source.py')]})
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'binding'}))


if __name__ == '__main__':
    main()

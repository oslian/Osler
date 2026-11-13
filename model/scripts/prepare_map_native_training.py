"""Freeze source-only native MAP transforms for the matched neural trial."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.availability_inputs import fit_numeric_rescale
from whole_body_jepa.history_event_input import prepare_events
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output = Path('artifacts/map_native_training_transforms_20260912.json')
    dest = root/'neural_transforms.npz'
    if output.exists() or dest.exists():
        raise ValueError('fresh outputs required')
    parent = Path('artifacts/map_native_query_history_complete_20260912.json')
    report = json.loads(parent.read_text())
    for name, expected in report['bindings'].items():
        if sha256(Path(name)) != expected:
            raise ValueError('changed dependency')
    q = pd.read_parquet(root/'queries.parquet')
    labels = np.load(root/'labels.npz')
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    n = len(variables)
    col = variables.index('map')
    supported = np.zeros((len(q), n), bool)
    def shards():
        for path in sorted((root/'input_shards').glob('shard_*.npz')):
            base = dict(np.load(path))
            for start in range(0, len(base['query_rows']), 128):
                selection = slice(start, start+128)
                observed = base['mask'][base['history'][base['query_rows'][selection]]].any(axis=1)
                supported[base['global_query_rows'][selection]] = observed
            yield base, dict(np.load(root/'native_query_shards'/path.name))
    storage = RaggedNativeBatch(shards(), len(q))
    saved, summaries, paths = {}, [], [parent, Path(__file__), root/'labels.npz']
    for seed in (7, 19, 31, 43, 59, 71, 83):
        path = root/f'partition_{seed}.npz'
        roles = dict(np.load(path))
        fit = roles['fit'] & labels['paired'][:, 0]
        subject = q['uniquepid'].to_numpy()
        fit_subjects = set(subject[fit])
        for role in ('tune', 'evaluation', 'hospital'):
            if fit_subjects.intersection(subject[roles[role]]):
                raise ValueError('patient overlap')
        indices = np.flatnonzero(fit)
        event_rows = np.unique(np.concatenate([np.array([0]),
            *[storage.sequences[i] for i in indices]]))
        arrays = dict(current=storage.input[:, :n], mask=storage.input[:, n:2*n],
            history=event_rows[None, :])
        center, scale = fit_numeric_rescale(arrays, np.array([0]), np.array([True]), col)
        nc, ns = np.zeros(n, np.float32), np.ones(n, np.float32)
        for name in ('heart_rate', 'map', 'o2sat', 'respiratory_rate', 'temperature'):
            j = variables.index(name)
            nc[j], ns[j] = center[j], scale[j]
        counts = pd.DataFrame(supported[fit]).groupby(subject[fit]).any().sum().to_numpy()
        _, ec, es = prepare_events(storage.event_values, event_rows)
        for name, value in dict(numeric_center=nc, numeric_scale=ns,
                               allowed=counts > 0, event_center=ec, event_scale=es).items():
            saved[f'{seed}_{name}'] = value
        summaries.append(dict(seed=seed, fit_queries=len(indices),
            fit_patients=len(fit_subjects), supported_variables=int((counts > 0).sum()),
            unique_fit_event_groups=len(event_rows)-1))
        paths.append(path)
        print('prepared', summaries[-1], flush=True)
    np.savez_compressed(dest, **saved)
    paths.extend([dest, Path('whole_body_jepa/ragged_native_batch.py'),
        Path('whole_body_jepa/availability_inputs.py'), Path('whole_body_jepa/history_event_input.py')])
    result = dict(complete=True, summaries=summaries, runtime_modified=False,
        training_performed=False, scope='source-only transforms, not fitted neural comparison',
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

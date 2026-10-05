"""Measure actual own/full active-update differences without using outcomes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output = Path('artifacts/map_native_update_counts_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    parent = Path('artifacts/map_native_query_training_summary_20260912.json')
    for name, expected in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'changed dependency: {name}')
    q = pd.read_parquet(root/'queries.parquet')
    def shards():
        for path in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(path)), dict(np.load(root/'native_query_shards'/path.name))
    raw = RaggedNativeBatch(shards(), len(q))
    transforms = np.load(root/'neural_transforms.npz')
    import torch
    ck = torch.load(root/'native_query_own_7.pt', map_location='cpu', weights_only=False)
    n = len(ck['variables'])
    col = ck['variables'].index('map')
    own_active = raw.input[:, n+col].astype(bool)
    own_count = np.array([own_active[s].sum() for s in raw.sequences])
    labels = np.load(root/'labels.npz')['paired'][:, 0]
    results = []
    for seed in (7,19,31,43,59,71,83):
        allowed = transforms[f'{seed}_allowed'].astype(bool)
        active = raw.input[:, n:2*n][:, allowed].astype(bool).any(axis=1)
        active |= np.isfinite(raw.event_values[:, 2])
        if active[0] or own_active[0]:
            raise ValueError('padding activates filter')
        if np.any(own_active & ~active):
            raise ValueError('full lost MAP update')
        counts = np.array([active[s].sum() for s in raw.sequences])
        roles = np.load(root/f'partition_{seed}.npz')
        summary = {}
        for role in ('fit','tune','evaluation','hospital'):
            keep = roles[role] & labels
            frame = pd.DataFrame({'subject':q.loc[keep, 'uniquepid'].to_numpy(),
                'own':own_count[keep], 'full':counts[keep],
                'extra':counts[keep]-own_count[keep],
                'has_extra':counts[keep]>own_count[keep]})
            patient = frame.groupby('subject').mean(numeric_only=True)
            summary[role] = dict(patients=len(patient), queries=int(keep.sum()),
                patient_equal_means={k:float(v) for k,v in patient.mean().items()})
        results.append(dict(seed=seed, roles=summary))
    result = dict(complete=True, results=results, training_performed=False,
        runtime_modified=False, scope='Input-only update cadence; not accuracy causation.',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),
            Path('whole_body_jepa/native_event_encoder.py'),
            root/'neural_transforms.npz')})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(results[0],indent=2))


if __name__ == '__main__':
    main()

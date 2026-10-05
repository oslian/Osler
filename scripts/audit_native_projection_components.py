"""Source-only observation-projection decomposition, not an accuracy claim."""
import json
from pathlib import Path

import numpy as np
import torch

from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    refpath = Path('artifacts/rate_glucose_jepa_20260911.json')
    output = Path('artifacts/native_projection_components_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(refpath.read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('changed input')
    native = np.load(root / 'native_tokens.npz')
    labels = np.load(root / 'multitarget_labels.npz')
    n = len(labels['variables']); k = list(labels['variables']).index('glucose')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        fit = np.load(root / f'partition_{seed}.npz')['fit'][valid] & labels['paired'][:, k]
        ptr, ids = native['query_ptr'], native['history_indices']
        selected = np.unique(np.concatenate([ids[ptr[i]:ptr[i+1]] for i in np.flatnonzero(fit)]))
        saved = torch.load(root / f'glucose_native_real_{seed}.pt', weights_only=True)
        allowed = np.asarray(saved['allowed_variables'], bool)
        x = native['input'][selected].copy()
        x[:, :n][:, ~allowed] = 0
        x[:, n:2*n][:, ~allowed] = 0
        # Match restrict_input_support's age convention after inspecting it.
        x[:, 2*n:3*n][:, ~allowed] = 7.
        event = native['event_values'][selected]
        normalized = np.where(np.isfinite(event), (event-saved['event_center'])/saved['event_scale'], 0)
        x = np.column_stack([x[:, :-1], normalized, np.isfinite(event), x[:, -1]])
        weight = saved['state_dict']['belief_filter.observation_projection.0.weight'].numpy()
        missing = x[:, n:2*n] == 0
        terms = {}
        for name in ('lab_values', 'missing_ages', 'observed_ages', 'observation_masks'):
            part = np.zeros_like(x)
            if name == 'lab_values': part[:, :n] = x[:, :n]
            if name == 'observation_masks': part[:, n:2*n] = x[:, n:2*n]
            if name == 'missing_ages': part[:, 2*n:3*n] = x[:, 2*n:3*n] * missing
            if name == 'observed_ages': part[:, 2*n:3*n] = x[:, 2*n:3*n] * ~missing
            projected = part @ weight.T
            terms[name] = dict(rms=float(np.sqrt(np.mean(projected**2))),
                centered_rms=float(np.sqrt(np.mean((projected-projected.mean(axis=0))**2))))
        results.append(dict(seed=seed, source_unique_tokens=len(selected), terms=terms))
    output.write_text(json.dumps(dict(complete=True, status='descriptive_only', runtime_modified=False,
        prediction_effect_measured=False, binding={str(p):sha256(p) for p in (refpath, Path(__file__))},
        seeds=results), indent=2)+'\n')
    for term in results[0]['terms']:
        print(term, {m: float(np.median([r['terms'][term][m] for r in results]))
                     for m in ('rms','centered_rms')})


if __name__ == '__main__': main()

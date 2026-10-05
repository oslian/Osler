"""Diagnose selection uncertainty without changing any selected predictor."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from whole_body_jepa.multisource_intervals import sha256


def main():
    report_path = Path('artifacts/native_reconstruction_nested_heads_20260917.json')
    output = Path('artifacts/native_reconstruction_nested_margins_20260917.json')
    if output.exists():
        raise ValueError('Diagnostic output already exists')
    report = json.loads(report_path.read_text())
    for name, digest in report['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Changed dependency: {name}')
    data = torch.load('data/research_recovery/site_raw_state_features_20260917.pt', weights_only=False)
    roles = torch.load('data/research_recovery/reconstruction_nested_selection_20260917.pt', weights_only=False)['roles']
    results = {}
    for target, row in report['targets'].items():
        choice = row['choice']
        if choice['status'] != 'development_candidate':
            results[target] = {'status': choice['status']}
            continue
        fit = [d for d in data if d['target'] == target and roles[d['subject_id']] == 'fit']
        tune = [d for d in data if d['target'] == target and roles[d['subject_id']] == 'readout_tune']
        if len({d['subject_id'] for d in tune}) != len(tune):
            raise ValueError('Bootstrap requires one row per patient-target')
        name = choice['selected']
        y = np.array([d['truth'] for d in tune])
        original = np.array([d['original'] for d in tune])
        if name == 'original':
            prediction = original.copy()
        elif name == 'population':
            prediction = np.full(len(tune), np.median([d['truth'] for d in fit]))
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(name.split('_')[1]), solver='svd'))
            model.fit(np.stack([d['body'] for d in fit]), [d['truth'] for d in fit])
            prediction = model.predict(np.stack([d['body'] for d in tune]))
        error = np.abs(prediction - y)
        if not np.isclose(error.mean(), choice['source_scores'][name], atol=1e-10, rtol=0):
            raise ValueError('Selected source score did not reproduce')
        delta = error - np.abs(original - y)
        rng = np.random.default_rng(20260917)
        samples = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(axis=1)
        ci = np.quantile(samples, [.025, .975]).tolist()
        external = row['comparisons']['original']['selected']
        results[target] = {
            'status': choice['status'], 'selected': name, 'selection_patients': len(tune),
            'selection_mae_delta': float(delta.mean()), 'selection_ci95_descriptive': ci,
            'evaluation_mae_delta': external['mean'], 'evaluation_ci95_nominal': external['ci95'],
        }
    switched = {t: r for t, r in results.items() if r.get('selected') not in (None, 'original')}
    summary = {
        'switched_targets': len(switched),
        'selection_ci_crosses_zero': [t for t, r in switched.items() if r['selection_ci95_descriptive'][1] >= 0],
        'selection_ci_below_zero_but_evaluation_harm': [t for t, r in switched.items() if r['selection_ci95_descriptive'][1] < 0 and r['evaluation_ci95_nominal'][0] > 0],
        'evaluation_harms': [t for t, r in switched.items() if r['evaluation_ci95_nominal'][0] > 0],
    }
    payload = dict(complete=True, targets=results, summary=summary,
                   bindings={str(report_path): sha256(report_path), str(Path(__file__)): sha256(Path(__file__))},
                   runtime_modified=False, promotion_allowed=False,
                   limitation='Selection intervals use the same data that selected the winner; they are descriptive and selection-biased, not an independent acceptance gate. No threshold is optimized using evaluation outcomes.')
    output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

"""Source-only head nomination followed by separate confirmation."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_1samp
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from whole_body_jepa.multisource_intervals import sha256


def evidence(delta, groups):
    delta = np.asarray(delta, dtype=float)
    site = np.array([delta[np.asarray(groups) == g].mean() for g in sorted(set(groups))])
    def pvalue(values):
        if len(values) < 2 or not np.isfinite(values).all():
            return 1.0
        if np.std(values) == 0:
            return 0.0 if values.mean() < 0 else 1.0
        return float(ttest_1samp(values, 0, alternative='less').pvalue)
    return dict(mean=float(delta.mean()), site_mean=float(site.mean()),
                patient_p=pvalue(delta), site_p=pvalue(site), sites=len(site))


def confirmed(items, patients):
    return patients >= 30 and all(
        r['sites'] >= 3 and r['mean'] < 0 and r['site_mean'] < 0
        and max(r['patient_p'], r['site_p']) < .05 / 39 for r in items.values())


def main():
    out = Path('artifacts/native_reconstruction_confirmed_heads_20260917.json')
    decisions = Path('artifacts/native_reconstruction_confirmed_choices_20260917.json')
    if out.exists() or decisions.exists():
        raise ValueError('Existing experiment')
    parent = Path('artifacts/native_reconstruction_nested_heads_20260917.json')
    previous = json.loads(parent.read_text())
    for path, digest in previous['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed dependency')
    cache = Path('data/research_recovery/site_raw_state_features_20260917.pt')
    split = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    data = torch.load(cache, weights_only=False)
    partition = torch.load(split, weights_only=False)
    roles, components = partition['roles'], partition['components']
    choices, predictions = {}, {}
    for target, old in previous['targets'].items():
        rows = {r: [d for d in data if d['target'] == target and roles[d['subject_id']] == r]
                for r in ('fit', 'readout_tune', 'evaluation')}
        if old['choice']['status'] != 'development_candidate':
            choices[target] = dict(accepted=False, status='insufficient_four_role_support')
            continue
        fit, tune = rows['fit'], rows['readout_tune']
        x = np.stack([d['body'] for d in fit]); y = np.array([d['truth'] for d in fit])
        groups = np.array([components[d['subject_id']] for d in fit])
        if len(set(groups)) < 5:
            raise ValueError('Insufficient nomination sites')
        candidates = [None, 1., 10., 100., 1000., 10000.]
        def model(alpha):
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
        scores = []
        for alpha in candidates:
            prediction = np.empty(len(y))
            for train, valid in GroupKFold(5).split(x, y, groups):
                prediction[valid] = np.median(y[train]) if alpha is None else model(alpha).fit(x[train], y[train]).predict(x[valid])
            scores.append(float(np.mean(abs(prediction-y))))
        alpha = candidates[int(np.argmin(scores))]
        fitted = None if alpha is None else model(alpha).fit(x, y)
        median = float(np.median(y))
        def predict(part):
            return np.full(len(part), median) if fitted is None else fitted.predict(np.stack([d['body'] for d in part]))
        ty = np.array([d['truth'] for d in tune]); p = predict(tune)
        tg = [components[d['subject_id']] for d in tune]
        refs = {'original': np.array([d['original'] for d in tune]), 'population': np.full(len(tune), median)}
        checks = {r: evidence(abs(p-ty)-abs(v-ty), tg) for r, v in refs.items()}
        accepted = confirmed(checks, len(tune))
        choices[target] = dict(alpha=alpha, nomination_scores=scores, confirmation=checks, accepted=accepted, status='development_only')
        predictions[target] = predict(rows['evaluation']) if accepted else np.array([d['original'] for d in rows['evaluation']])
    bindings = {str(p): sha256(p) for p in (Path(__file__), parent, cache, split, Path('RECONSTRUCTION_CONFIRMED_HEAD_PROTOCOL_20260917.md'))}
    decisions.write_text(json.dumps(dict(choices=choices, bindings=bindings), indent=2)+'\n')
    results = {}
    for target, old in previous['targets'].items():
        rows = [d for d in data if d['target'] == target and roles[d['subject_id']] == 'evaluation']
        y = np.array([d['truth'] for d in rows]); original = np.array([d['original'] for d in rows])
        selected = predictions.get(target, original)
        if not np.isclose(np.mean(abs(original-y)), old['mae']['original'], rtol=0, atol=1e-10):
            raise ValueError('Evaluation reproduction failed')
        results[target] = dict(original_mae=float(np.mean(abs(original-y))), selected_mae=float(np.mean(abs(selected-y))))
    payload = dict(complete=True, choices=choices, results=results, unsupported=previous['unsupported'], bindings=bindings,
                   runtime_modified=False, promotion_allowed=False, decisions_sha256=sha256(decisions))
    out.write_text(json.dumps(payload, indent=2)+'\n')
    print('Accepted development heads:', [t for t, c in choices.items() if c['accepted']])
    print('Macro original/selected:', *(np.mean([r[k] for r in results.values()]) for k in ('original_mae', 'selected_mae')))


if __name__ == '__main__':
    main()

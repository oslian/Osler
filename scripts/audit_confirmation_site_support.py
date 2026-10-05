"""Resolve hospital-support uncertainty in fixed nominated heads."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from whole_body_jepa.multisource_intervals import sha256


def main():
    source = Path('artifacts/native_reconstruction_confirmed_heads_20260917.json')
    output = Path('artifacts/native_reconstruction_confirmation_support_20260917.json')
    if output.exists():
        raise ValueError('Existing diagnostic')
    report = json.loads(source.read_text())
    for path, digest in report['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed dependency')
    data = torch.load('data/research_recovery/site_raw_state_features_20260917.pt', weights_only=False)
    split = torch.load('data/research_recovery/reconstruction_nested_selection_20260917.pt', weights_only=False)
    result = {}
    for target, choice in report['choices'].items():
        if 'alpha' not in choice:
            result[target] = {'status': choice['status']}
            continue
        fit = [d for d in data if d['target'] == target and split['roles'][d['subject_id']] == 'fit']
        rows = [d for d in data if d['target'] == target and split['roles'][d['subject_id']] == 'readout_tune']
        y = np.array([d['truth'] for d in fit])
        alpha = choice['alpha']
        if alpha is None:
            pred = np.full(len(rows), np.median(y))
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
            model.fit(np.stack([d['body'] for d in fit]), y)
            pred = model.predict(np.stack([d['body'] for d in rows]))
        truth = np.array([d['truth'] for d in rows])
        delta = abs(pred-truth)-abs(np.array([d['original'] for d in rows])-truth)
        group = np.array([split['components'][d['subject_id']] for d in rows])
        sites = []
        for g in sorted(set(group)):
            d = delta[group == g]
            rng = np.random.default_rng(20260917)
            ci = np.quantile(d[rng.integers(len(d), size=(5000, len(d)))].mean(axis=1), [.025, .975]).tolist()
            sites.append(dict(patients=len(d), delta=float(d.mean()), ci95_descriptive=ci,
                              insufficient_for_existing_30_patient_standard=len(d)<30))
        sites.sort(key=lambda r: (r['patients'], r['delta']))
        if not np.isclose(delta.mean(), choice['confirmation']['original']['mean'], rtol=0, atol=1e-10):
            raise ValueError('Score reproduction failed')
        result[target] = dict(sites=sites, supported_sites=sum(s['patients']>=30 for s in sites),
                              pooled_delta=float(delta.mean()), equal_site_delta=float(np.mean([s['delta'] for s in sites])))
    output.write_text(json.dumps(dict(complete=True,targets=result,unsupported=report['unsupported'],
        bindings={str(source):sha256(source),str(Path(__file__)):sha256(Path(__file__))},
        promotion_allowed=False,runtime_modified=False,
        limitation='Per-site CIs descriptive and unadjusted; tiny sites do not establish site-specific harm. No exclusions or gate changes.'),indent=2)+'\n')
    for t in ('bun','phosphate','bicarbonate'):
        print(t, json.dumps(result[t]))


if __name__ == '__main__':
    main()

"""Decompose frozen continuation errors; never select or alter a predictor."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.audit_confirmed_reconstruction_heads import evidence
from whole_body_jepa.multisource_intervals import sha256


def site_summary(delta, sites):
    delta = np.asarray(delta, dtype=float)
    sites = np.asarray(sites)
    if not len(delta) or delta.shape != sites.shape or not np.isfinite(delta).all():
        raise ValueError('Invalid paired site evidence')
    groups = [delta[sites == s] for s in sorted(set(sites))]
    means = np.array([v.mean() for v in groups])
    weights = np.array([len(v) for v in groups], dtype=float) / len(delta)
    loo = [float(np.delete(means, i).mean()) for i in range(len(means))] if len(means) > 1 else []
    return dict(pooled=float(delta.mean()), equal_site=float(means.mean()),
                sites=len(groups), improving_sites=int((means < 0).sum()),
                worsening_sites=int((means > 0).sum()),
                minimum_site_patients=min(map(len, groups)),
                effective_sites=float(1 / np.square(weights).sum()),
                weighting_sign_disagreement=bool((delta.mean() < 0) != (means.mean() < 0)),
                leave_one_site_out_range=[min(loo), max(loo)] if loo else None,
                all_leave_one_site_out_improve=bool(loo and max(loo) < 0))


def main():
    parent = Path('artifacts/native_reconstruction_coverage_evaluation_20260920.json')
    out = Path('artifacts/native_reconstruction_continued_site_transfer_20260920.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    report = json.loads(parent.read_text())
    for p, digest in report['bindings'].items():
        if sha256(Path(p)) != digest:
            raise ValueError(f'Changed binding: {p}')
    results = {}
    for group, targets in report['groups'].items():
        path = Path(f'data/research_recovery/broad_source_20260917/coverage_evaluation/{group}.pt')
        rows = torch.load(path, weights_only=False)
        keys = [(r['patient'], r['target']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate patient-target')
        per = {}
        for target, previous in targets.items():
            if 'mae' not in previous:
                per[target] = dict(status='untrained')
                continue
            selected = [r for r in rows if r['target'] == target]
            checks = {}
            for reference, original in previous['comparisons'].items():
                delta = [abs(r['continued_real'] - r['truth']) - abs(r[reference] - r['truth']) for r in selected]
                sites = [r['site'] for r in selected]
                reproduced = evidence(delta, sites)
                if any(not np.isclose(reproduced[k], original[k], rtol=0, atol=1e-10) for k in original):
                    raise ValueError('Gate evidence not reproduced')
                reasons = []
                for name, condition in (
                    ('mean_not_better', original['mean'] >= 0),
                    ('site_mean_not_better', original['site_mean'] >= 0),
                    ('patient_evidence', original['patient_p'] >= .05 / 39),
                    ('site_evidence', original['site_p'] >= .05 / 39),
                    ('site_support', original['sites'] < 3)):
                    if condition:
                        reasons.append(name)
                checks[reference] = dict(**site_summary(delta, sites), failures=reasons)
            per[target] = dict(comparisons=checks, source_supported=previous['source_supported'],
                               core_gate=previous['core_gate'], full_gate=previous['development_gate'])
        results[group] = per
        for reference in ('old', 'expanded_real', 'continued_placebo'):
            counts = Counter(reason for r in per.values() if 'comparisons' in r
                             for reason in r['comparisons'][reference]['failures'])
            swaps = sum(r['comparisons'][reference]['weighting_sign_disagreement'] for r in per.values() if 'comparisons' in r)
            print(group, reference, dict(counts), 'weighting_sign_changes', swaps, flush=True)
    out.write_text(json.dumps(dict(groups=results, runtime_modified=False, promotion_allowed=False,
        bindings={str(p): sha256(p) for p in (parent, Path(__file__))},
        limitation='Descriptive reused-cohort site analysis. LOO is sensitivity, not new validation or causal attribution. No gate changes.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

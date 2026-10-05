"""Descriptive decomposition; no fitting, selection or promotion."""
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def decompose(truth, old, new, old_center, new_center):
    y = np.asarray(truth)
    eo, en = abs(np.asarray(old)-y), abs(np.asarray(new)-y)
    po, pn = abs(np.asarray(old_center)-y), abs(np.asarray(new_center)-y)
    prior = pn-po
    increment = (en-pn)-(eo-po)
    if not np.allclose(en-eo, prior+increment, atol=1e-12):
        raise ValueError('Error decomposition failed')
    return dict(total=float((en-eo).mean()), prior=float(prior.mean()),
                incremental_change=float(increment.mean()),
                old_increment=float((eo-po).mean()),
                new_increment=float((en-pn).mean()))


def load_rows(index):
    rows = {}
    for item in torch.load(index, weights_only=False):
        path = Path(item['path'])
        if sha256(path) != item['sha256']:
            raise ValueError('Prediction hash changed')
        for r in torch.load(path, weights_only=False):
            key = (r['patient'], r['target'])
            if key in rows:
                raise ValueError('Duplicate patient-target')
            rows[key] = r
    return rows


def main():
    output = Path('artifacts/native_reconstruction_broad_prior_increment_20260917.json')
    if output.exists():
        raise ValueError('Existing audit')
    report = Path('artifacts/native_reconstruction_matched_broad_evaluation_20260917.json')
    parent = json.loads(report.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Parent binding changed')
    oi = Path('data/research_recovery/expanded_confirmation_20260917/predictions/index.pt')
    ni = Path('data/research_recovery/broad_source_20260917/matched_evaluation/index.pt')
    old, new = load_rows(oi), load_rows(ni)
    if old.keys() != new.keys():
        raise ValueError('Different pairs')
    results = {}
    for t, summary in parent['targets'].items():
        keys = [k for k in new if k[1] == t]
        if not keys:
            results[t] = dict(status='no_trained_source_head')
            continue
        for k in keys:
            if old[k]['truth'] != new[k]['truth'] or old[k]['site'] != new[k]['site']:
                raise ValueError('Pair mismatch')
        def score(ks):
            return decompose([new[k]['truth'] for k in ks],
                             [old[k]['original'] for k in ks],
                             [new[k]['real'] for k in ks],
                             [old[k]['population'] for k in ks],
                             [new[k]['population'] for k in ks])
        overall = score(keys)
        expected = summary['mae']['real']-summary['mae']['old']
        if not np.isclose(overall['total'], expected, atol=1e-10, rtol=0):
            raise ValueError('Score mismatch')
        sites = [score([k for k in keys if new[k]['site'] == s])
                 for s in sorted({new[k]['site'] for k in keys})]
        results[t] = dict(patients=len(keys), decomposition=overall,
                         sites=len(sites),
                         sites_new_beats_center=sum(s['new_increment']<0 for s in sites),
                         sites_increment_improves=sum(s['incremental_change']<0 for s in sites),
                         equal_site={m:float(np.mean([s[m] for s in sites])) for m in overall})
    output.write_text(json.dumps(dict(targets=results, descriptive_only=True,
        limitation='Algebraic error decomposition, not causal attribution. Different cohort fitting changes weights and priors together.',
        bindings={str(p):sha256(p) for p in (Path(__file__),report,oi,ni)},
        runtime_modified=False, promotion_allowed=False), indent=2)+'\n')
    for t,r in results.items():
        if 'decomposition' in r:
            print(t, r['decomposition'], 'sites', r['sites_new_beats_center'], '/', r['sites'])


if __name__ == '__main__':
    main()

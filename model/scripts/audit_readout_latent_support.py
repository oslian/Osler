"""Fit-only latent support diagnosis; no support-based deployment rule."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
from whole_body_jepa.multisource_intervals import sha256


def support_distances(train, query):
    mean = train.mean(axis=0)
    scale = train.std(axis=0); scale[scale < 1e-8] = 1.
    z = (train-mean)/scale
    model = LedoitWolf().fit(z)
    fitting = model.mahalanobis(z)
    threshold = float(np.quantile(fitting, .99))
    return model.mahalanobis((query-mean)/scale), threshold


def strata(delta, distance, threshold):
    result = {}
    for name, mask in (('inside', distance <= threshold), ('outside', distance > threshold)):
        result[name] = dict(patients=int(mask.sum()),
                            mean_delta=float(delta[mask].mean()) if mask.any() else None)
    return result


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    pp = Path('artifacts/native_reconstruction_anchored_readout_20260920.json')
    sp = Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    out = Path('artifacts/native_reconstruction_readout_latent_support_20260920.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    parent = json.loads(pp.read_text())
    for path, digest in parent['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Parent changed')
    mp = root/'site_balance_manifest_20260920.pt'
    cp = root/'anchored_readout_20260920/candidate.pt'
    if sha256(cp) != parent['candidate_sha256']:
        raise ValueError('Candidate changed')
    candidate = torch.load(cp, weights_only=False)['state_dict']
    manifest = torch.load(mp, weights_only=False)
    variables = torch.load(root/'joint_source_manifest.pt', weights_only=False)['variables']
    data = {role: {t: {g: [] for g in ('selected', 'broad')} for t in parent['choices']} for role in ('fit', 'encoder_tune')}
    digest = hashlib.sha256()
    for i, s in enumerate(manifest['shards']):
        path = root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt'
        digest.update((path.name+sha256(path)).encode())
        saved = torch.load(path, weights_only=False)
        if saved['shard'] != s:
            raise ValueError('Wrong feature identity')
        for row in saved['rows']:
            data[s['role']][row['target']][s['group']].append(row)
    if digest.hexdigest() != json.loads(sp.read_text())['cache_digest']:
        raise ValueError('Feature cache changed')
    results = {}
    with threadpool_limits(limits=4):
        for target, fitgroups in data['fit'].items():
            col = variables.index(target); per = {}
            for group, rows in fitgroups.items():
                held = data['encoder_tune'][target][group]
                train = np.stack([r['body'] for r in rows]).astype(float)
                x = np.stack([r['body'] for r in held])
                y = np.array([r['truth'] for r in held]); old = np.array([r['prediction'] for r in held])
                new = old.copy() if parent['choices'][target]['name']=='unchanged' else x@candidate['readout.weight'][col].numpy()+float(candidate['readout.bias'][col])
                delta = abs(new-y)-abs(old-y)
                expected = parent['source_selection'][target][group]
                if not np.isclose(delta.mean(), expected['after']-expected['before'], atol=1e-10, rtol=0):
                    raise ValueError('Parent errors differ')
                distance, threshold = support_distances(train, x.astype(float))
                per[group] = dict(fit_patients=len(rows), held_patients=len(held), threshold=threshold,
                    outside_fraction=float(np.mean(distance>threshold)), mean_delta=float(delta.mean()),
                    strata=strata(delta, distance, threshold))
            results[target] = per
    harmful = [(t, g, v) for t, per in results.items() for g, v in per.items() if v['mean_delta']>0]
    summary = dict(targets=len(results), harmful_groups=len(harmful),
        harmful_groups_with_supported_inside_harm=[dict(target=t, group=g, patients=v['strata']['inside']['patients'], mean_delta=v['strata']['inside']['mean_delta'])
            for t, g, v in harmful if v['strata']['inside']['patients']>=30 and v['strata']['inside']['mean_delta']>0],
        median_outside_fraction={g:float(np.median([per[g]['outside_fraction'] for per in results.values()])) for g in ('selected','broad')})
    out.write_text(json.dumps(dict(targets=results, summary=summary, cache_digest=digest.hexdigest(),
        bindings={str(p):sha256(p) for p in (Path(__file__), pp, sp, mp, cp)},
        runtime_modified=False, promotion_allowed=False,
        limitation='Within-source-group StandardScaler-equivalent plus LedoitWolf, fit-only empirical 99th percentile squared distance. In-sample threshold is not calibrated novelty coverage. Descriptive one patient-target pair; inside ellipsoid does not prove identical physiology or covariate overlap. No outcome-based gate or causal claim.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

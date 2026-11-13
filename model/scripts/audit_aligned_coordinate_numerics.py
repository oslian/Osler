"""Check saved observation-aligned mappings without refitting or choosing models."""
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_aligned_coordinates import ARMS, DEST, transform
from whole_body_jepa.multisource_intervals import sha256


def check_mapping(mapping):
    scalers = {name: check_scaler(mapping[name])
               for name in ('body_scaler', 'raw_scaler')}
    for name, width in (('body_scaler', 64), ('raw_scaler', 265)):
        if np.asarray(mapping[name].mean_).shape != (width,):
            raise ValueError('Wrong scaler width')
    constant = np.asarray(mapping['constant'])
    if constant.shape != (265,) or constant.dtype != np.dtype(bool):
        raise ValueError('Invalid constant mask')
    for arm in ('aligned_body', 'shuffled_axes'):
        model = mapping[arm]
        coef, intercept = np.asarray(model.coef_), np.asarray(model.intercept_)
        if (coef.shape != (265, 64) or intercept.shape != (265,)
                or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
            raise ValueError('Invalid ridge coefficients')
    basis = np.asarray(mapping['random_axes'])
    if (basis.shape != (64, 265) or not np.isfinite(basis).all()
            or not np.allclose(np.linalg.norm(basis, axis=0), 1., atol=1e-12, rtol=1e-12)):
        raise ValueError('Invalid random unit basis')
    bs = mapping['body_scaler']
    probes = bs.mean_[None, :] + np.array([-1., 0., 1.])[:, None] * bs.scale_[None, :]
    for arm in ARMS:
        z = transform(probes, mapping, arm)
        if z.shape != (3, 329) or not np.array_equal(z[:, :64], probes):
            raise ValueError('Original body coordinates changed')
        if arm != 'random_axes' and np.any(z[:, 64:][:, constant] != 0):
            raise ValueError('Constant decoded coordinates are not zero')
    return dict(scalers=scalers, constant_outputs=int(constant.sum()),
                coefficients_finite=True, original_body_preserved=True,
                constant_decoded_outputs_zero=True, synthetic_probe_outputs_finite=True)


def main():
    out = Path('artifacts/native_reconstruction_aligned_coordinates_numerics_20261003.json')
    if out.exists():
        raise ValueError('Existing numerical audit')
    paths = [Path(__file__), Path('scripts/audit_scaled_kernel_numerics.py'),
             Path('scripts/train_aligned_coordinates.py')]
    folds, checked, expected = {}, {}, None
    for f in range(5):
        rp = Path(f'artifacts/native_reconstruction_aligned_coordinates_fold{f}_20261003.json')
        report = json.loads(rp.read_text())
        verify_bound_files(report['bindings'], checked)
        targets = sorted(report['targets'])
        if report['fold'] != f or len(targets) != 47 or (expected is not None and targets != expected):
            raise ValueError('Incomplete or inconsistent target family')
        expected = targets
        cp = DEST / f'fold{f}/contract.json'
        contract = json.loads(cp.read_text())
        if contract['bindings'] != report['bindings'] or contract['fold'] != f:
            raise ValueError('Changed contract')
        paths.extend((rp, cp))
        folds[str(f)] = {}
        for t in targets:
            meta = report['targets'][t]
            mp = DEST / f'fold{f}/{t}.joblib'
            if (meta['fold'] != f or meta['target'] != t
                    or meta['model_sha256'] != sha256(mp)
                    or meta['contract_sha256'] != sha256(cp)):
                raise ValueError('Changed fitted artifact')
            saved = joblib.load(mp)
            for arm in ARMS:
                head = saved['models'][arm]
                if head.n_iter_ != 100 or head.n_features_in_ != 329:
                    raise ValueError('Incomplete or wrong-width head')
            folds[str(f)][t] = check_mapping(saved['mapping'])
            paths.append(mp)
        print('Numerics verified fold', f, '47 mappings', flush=True)
    scalers = [s for fold in folds.values() for t in fold.values() for s in t['scalers'].values()]
    summary = dict(mappings=235, scalers=len(scalers), heads=705,
                   negative_variance_scalers=sum(s['negative_variances'] > 0 for s in scalers),
                   negative_variances=sum(s['negative_variances'] for s in scalers),
                   minimum_variance=min(s['minimum_variance'] for s in scalers),
                   all_checks_passed=True)
    out.write_text(json.dumps(dict(summary=summary, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False,
        limitation='Numerical diagnostic only. Synthetic probes check arithmetic, not physiological validity or accuracy. '
                   'Negative variances are accepted only within the existing roundoff bound with scale exactly one. '
                   'All-query identities and finite predictions are checked separately by score_aligned_coordinates.'),
        indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

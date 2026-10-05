"""Audit E20 saved readers and unchanged E19 baselines without retraining."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.probe_assimilation_content import DEST, PARENT, ARMS
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_reconstruction_assimilation_verification_20261003.json')
    if output.exists():
        raise ValueError('Verification already exists')
    ap = Path('artifacts/native_reconstruction_assimilation_assessment_20261003.json')
    assessment = json.loads(ap.read_text())
    checked = {}

    def verify(path, expected):
        path = Path(path)
        actual = checked.setdefault(str(path), sha256(path))
        if actual != expected:
            raise ValueError(f'Binding mismatch: {path}')

    for path, digest in assessment['bindings'].items():
        verify(path, digest)
    frames, scalers, replay = [], [], []
    with threadpool_limits(limits=4):
        for fold in range(5):
            report = json.loads(Path(f'artifacts/native_reconstruction_assimilation_fold{fold}_20261003.json').read_text())
            contract = report['contract']
            pp = Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
            verify(pp, contract['parent_sha256'])
            parent = json.loads(pp.read_text())
            assert contract['parent_contract'] == parent['contract']
            for group in (contract['bindings'], contract['parent_contract']['bindings']):
                for path, digest in group.items():
                    verify(path, digest)
            variables = parent['contract']['variables']
            assert len(variables) == 53 and set(report['targets']) == set(variables)
            dest, old = DEST/f'fold{fold}', PARENT/f'fold{fold}'
            cp = dest/'contract.json'
            assert json.loads(cp.read_text()) == contract
            data, stages = {}, {}
            for role, meta in report['extraction'].items():
                verify(cp, meta['contract_sha256'])
                verify(dest/f'{role}.joblib', meta['sha256'])
                verify(old/f'{role}.joblib', parent['extraction'][role]['sha256'])
                stages[role] = joblib.load(dest/f'{role}.joblib')
                data[role] = joblib.load(old/f'{role}.joblib')
                assert stages[role].shape == (len(data[role]['patient']), 53, 64)
                assert np.isfinite(stages[role]).all()
                replay.append(meta)
            assert not set(data['fit']['patient']) & set(data['selection']['patient'])
            for c, variable in enumerate(variables):
                meta = report['targets'][variable]
                prior = parent['targets'][variable]
                assert (meta['status'], meta['support']) == (prior['status'], prior['support'])
                if meta['status'] != 'fitted':
                    continue
                verify(cp, meta['contract_sha256'])
                verify(dest/f'{variable}.joblib', meta['model_sha256'])
                verify(dest/f'{variable}.parquet', meta['prediction_sha256'])
                verify(old/f'{variable}.parquet', prior['prediction_sha256'])
                frame = pd.read_parquet(dest/f'{variable}.parquet')
                baseline = pd.read_parquet(old/f'{variable}.parquet')
                pd.testing.assert_frame_equal(frame[baseline.columns], baseline)
                q = data['selection']
                m = q['raw'][:, 106+c].astype(bool)
                assert frame.patient.tolist() == q['patient'][m].tolist()
                assert frame.component.tolist() == q['component'][m].tolist()
                assert np.array_equal(frame.truth, q['raw'][m, c])
                assert np.array_equal(frame.context, q['context'][m])
                assert (frame.fold == fold).all() and (frame.variable == variable).all()
                assert np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()
                models = joblib.load(dest/f'{variable}.joblib')
                numeric = check_scaler(models['scaler'])
                assert numeric == meta['numerics']
                scalers.append(numeric)
                x = np.c_[stages['selection'][m, c], q['raw'][m, 106:]]
                assert x.shape[1] == 223 and models['hgb'].n_iter_ == 100
                assert np.isfinite(models['ridge'].coef_).all()
                assert np.isfinite(models['ridge'].intercept_).all()
                np.testing.assert_allclose(models['ridge'].predict(models['scaler'].transform(x)),
                                           frame.ridge_post, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(models['hgb'].predict(x), frame.hgb_post, atol=1e-12, rtol=1e-12)
                frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary = dict(variable_folds=len(frames), new_readers=len(frames)*2,
        query_occurrences=len(rows), patients=int(rows.patient.nunique()),
        variables=int(rows.variable.nunique()), scalers=len(scalers),
        negative_variance_scalers=sum(s['negative_variances']>0 for s in scalers),
        minimum_variance=min(s['minimum_variance'] for s in scalers),
        max_event_reproduction_error=max(r['max_event_error'] for r in replay),
        max_final_reproduction_error=max(r['max_final_error'] for r in replay),
        max_batch_shape_difference=max(r['max_event_batch_error'] for r in replay),
        original_predictions_unchanged=True, saved_predictions_reproduced=True,
        finite_positive_scales=True)
    for key in ('variable_folds', 'query_occurrences', 'patients', 'variables'):
        assert summary[key] == assessment['summary'][key]
    output.write_text(json.dumps(dict(summary=summary, checked_files=len(checked),
        bindings={str(ap): sha256(ap), str(Path(__file__)): sha256(Path(__file__))},
        runtime_modified=False, promotion_allowed=False), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

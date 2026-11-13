"""Verify the completed observed-content experiment without fitting models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.probe_observed_content import ARMS, DEST
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_reconstruction_observed_content_verification_20261003.json')
    if output.exists():
        raise ValueError('Completed verification already exists')
    assessment_path = Path('artifacts/native_reconstruction_observed_content_assessment_20261003.json')
    assessment = json.loads(assessment_path.read_text())
    checked = {}

    def verify(path, expected):
        path = str(path)
        if path not in checked:
            checked[path] = sha256(Path(path))
        if checked[path] != expected:
            raise ValueError(f'Binding mismatch: {path}')

    for path, digest in assessment['bindings'].items():
        verify(path, digest)
    scalers, frames, support = [], [], {}
    fitted = 0
    with threadpool_limits(limits=4):
        for fold in range(5):
            report_path = Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
            report = json.loads(report_path.read_text())
            contract = report['contract']
            for path, digest in contract['bindings'].items():
                verify(path, digest)
            variables = contract['variables']
            assert len(variables) == 53 and set(report['targets']) == set(variables)
            dest = DEST / f'fold{fold}'
            cp = dest / 'contract.json'
            assert json.loads(cp.read_text()) == contract
            data = {}
            for role, meta in report['extraction'].items():
                verify(cp, meta['contract_sha256'])
                verify(dest / f'{role}.joblib', meta['sha256'])
                data[role] = joblib.load(dest / f'{role}.joblib')
                assert len(set(data[role]['patient'])) == meta['patients']
            assert not set(data['fit']['patient']) & set(data['selection']['patient'])
            f, q = data['fit'], data['selection']
            support[str(fold)] = {}
            for c, variable in enumerate(variables):
                meta = report['targets'][variable]
                fm, qm = f['raw'][:, 106+c].astype(bool), q['raw'][:, 106+c].astype(bool)
                counts = dict(fit=int(fm.sum()), selection=int(qm.sum()),
                              components=len(set(q['component'][qm])))
                assert counts == meta['support']
                eligible = (counts['fit'] >= 50 and counts['selection'] >= 30
                            and counts['components'] >= 3 and np.ptp(f['raw'][fm, c]) > 0)
                assert eligible == (meta['status'] == 'fitted')
                support[str(fold)][variable] = dict(status=meta['status'], **counts)
                if not eligible:
                    continue
                verify(cp, meta['contract_sha256'])
                verify(dest / f'{variable}.joblib', meta['model_sha256'])
                verify(dest / f'{variable}.parquet', meta['prediction_sha256'])
                models = joblib.load(dest / f'{variable}.joblib')
                frame = pd.read_parquet(dest / f'{variable}.parquet')
                assert frame.patient.tolist() == q['patient'][qm].tolist()
                assert frame.component.tolist() == q['component'][qm].tolist()
                assert np.array_equal(frame.context, q['context'][qm])
                assert np.array_equal(frame.truth, q['raw'][qm, c])
                assert (frame.fold == fold).all() and (frame.variable == variable).all()
                assert np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()
                process = q['raw'][qm, 106:]
                for source in ('event', 'body', 'process'):
                    model = models[source]
                    scalers.append(check_scaler(model['scaler']))
                    x = process if source == 'process' else np.c_[
                        q['event'][qm, c] if source == 'event' else q['body'][qm], process]
                    assert x.shape[1] == (159 if source == 'process' else 223)
                    assert model['hgb'].n_iter_ == 100
                    assert np.isfinite(model['ridge'].coef_).all()
                    assert np.isfinite(model['ridge'].intercept_).all()
                    np.testing.assert_allclose(model['ridge'].predict(model['scaler'].transform(x)),
                                               frame['ridge_'+source], atol=1e-10, rtol=1e-10)
                    np.testing.assert_allclose(model['hgb'].predict(x), frame['hgb_'+source],
                                               atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(frame['median'], np.full(len(frame), np.median(f['raw'][fm, c])))
                fitted += 1
                frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary = dict(fitted_variable_folds=fitted, trained_readers=fitted*6,
                   query_occurrences=len(rows), patients=int(rows.patient.nunique()),
                   scalers=len(scalers), negative_variance_scalers=sum(s['negative_variances'] > 0 for s in scalers),
                   minimum_variance=min(s['minimum_variance'] for s in scalers),
                   finite_positive_scales=True, saved_predictions_reproduced=True)
    assert len(rows) == assessment['summary']['query_occurrences']
    assert summary['patients'] == assessment['summary']['patients']
    output.write_text(json.dumps(dict(summary=summary, support=support,
        bindings={str(assessment_path): sha256(assessment_path), str(Path(__file__)): sha256(Path(__file__)),
                  'scripts/audit_scaled_kernel_numerics.py': sha256(Path('scripts/audit_scaled_kernel_numerics.py'))},
        checked_files=len(checked), runtime_modified=False, promotion_allowed=False), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

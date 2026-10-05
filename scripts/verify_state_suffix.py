"""Reload all E22 readers, preserve baselines, and recompute paired assessment."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.probe_state_suffix import ARMS, ASSIMILATION, DEST, OBSERVED, summarize
from whole_body_jepa.multisource_intervals import sha256


def numeric_summary_parity(actual, expected):
    """Permit reduction roundoff only in explicitly selected numeric summaries."""
    if isinstance(actual, dict):
        assert set(actual) == set(expected)
        return max((numeric_summary_parity(actual[k], expected[k]) for k in actual), default=0.)
    assert isinstance(actual, float) and isinstance(expected, float)
    assert np.isfinite([actual, expected]).all()
    error = abs(actual-expected)
    assert error <= 64*np.finfo(np.float64).eps*max(1., abs(actual), abs(expected))
    return error


def assessment_parity(targets, summary, expected):
    assert set(targets) == set(expected['targets'])
    errors = []
    for v, result in targets.items():
        prior = expected['targets'][v]
        assert set(result) == set(prior)
        for key in result:
            if key in ('normalized_mae', 'descriptive_interaction'):
                errors.append(numeric_summary_parity(result[key], prior[key]))
            else:
                assert result[key] == prior[key]
    assert set(summary) == set(expected['summary'])
    for key in summary:
        if key == 'normalized_macro':
            errors.append(numeric_summary_parity(summary[key], expected['summary'][key]))
        else:
            assert summary[key] == expected['summary'][key]
    return max(errors, default=0.)


def main():
    output = Path('artifacts/native_reconstruction_suffix_verification_20261003.json')
    if output.exists(): raise ValueError('Verification exists')
    ap = Path('artifacts/native_reconstruction_suffix_assessment_20261003.json')
    assessment = json.loads(ap.read_text())
    checked = {}

    def verify(path, expected):
        path = Path(path)
        if str(path) not in checked: checked[str(path)] = sha256(path)
        if checked[str(path)] != expected: raise ValueError(f'Binding mismatch: {path}')

    for p, h in assessment['bindings'].items(): verify(p, h)
    frames, scalers, parity = [], [], []
    with threadpool_limits(limits=4):
        for fold in range(5):
            report = json.loads(Path(f'artifacts/native_reconstruction_suffix_fold{fold}_20261003.json').read_text())
            contract = report['contract']
            pp = Path(f'artifacts/native_reconstruction_assimilation_fold{fold}_20261003.json')
            verify(pp, contract['parent_sha256'])
            parent = json.loads(pp.read_text())
            op = Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
            verify(op, parent['contract']['parent_sha256'])
            observed = json.loads(op.read_text())
            for bindings in (contract['bindings'], contract['encoded_contract']['bindings'],
                             contract['encoded_contract']['parent']['bindings'], parent['contract']['bindings']):
                for p, h in bindings.items(): verify(p, h)
            assert contract['encoded_contract']['parent'] == observed['contract'] == parent['contract']['parent_contract']
            variables = observed['contract']['variables']
            assert len(variables) == 53 and set(report['targets']) == set(variables)
            dest, old = DEST/f'fold{fold}', ASSIMILATION/f'fold{fold}'
            cp = dest/'contract.json'
            assert json.loads(cp.read_text()) == contract
            data, stages = {}, {}
            for role, meta in report['extraction'].items():
                verify(cp, meta['contract_sha256']); verify(dest/f'{role}.joblib', meta['sha256'])
                dp = OBSERVED/f'fold{fold}/{role}.joblib'
                verify(dp, observed['extraction'][role]['sha256'])
                data[role], stages[role] = joblib.load(dp), joblib.load(dest/f'{role}.joblib')
                assert set(stages[role]) == {'advance', 'update'}
                for stage in stages[role].values():
                    assert stage.shape == (len(data[role]['patient']), 53, 64) and np.isfinite(stage).all()
                assert meta['post_max_error'] <= 1e-6 and meta['final_max_error'] <= 1e-6
                parity.append(meta)
            assert not set(data['fit']['patient']) & set(data['selection']['patient'])
            for c, v in enumerate(variables):
                meta, prior = report['targets'][v], parent['targets'][v]
                assert (meta['status'], meta['support']) == (prior['status'], prior['support'])
                if meta['status'] != 'fitted': continue
                verify(cp, meta['contract_sha256'])
                verify(dest/f'{v}.joblib', meta['model_sha256'])
                verify(dest/f'{v}.parquet', meta['prediction_sha256'])
                verify(old/f'{v}.parquet', prior['prediction_sha256'])
                frame, baseline = pd.read_parquet(dest/f'{v}.parquet'), pd.read_parquet(old/f'{v}.parquet')
                pd.testing.assert_frame_equal(frame[baseline.columns], baseline)
                q = data['selection']; mask = q['raw'][:, 106+c].astype(bool)
                assert frame.patient.tolist() == q['patient'][mask].tolist()
                assert frame.component.tolist() == q['component'][mask].tolist()
                np.testing.assert_array_equal(frame.context, q['context'][mask])
                np.testing.assert_array_equal(frame.truth, q['raw'][mask, c])
                assert (frame.fold == fold).all() and (frame.variable == v).all()
                assert np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()
                models = joblib.load(dest/f'{v}.joblib')
                for arm in ('advance', 'update'):
                    m = models[arm]
                    numeric = check_scaler(m['scaler'])
                    assert numeric == meta['numerics'][arm]
                    scalers.append(numeric)
                    x = np.c_[stages['selection'][arm][mask, c], q['raw'][mask, 106:]]
                    assert x.shape[1] == 223 and m['hgb'].n_iter_ == 100
                    np.testing.assert_allclose(m['ridge'].predict(m['scaler'].transform(x)),
                                               frame['ridge_'+arm], atol=1e-10, rtol=1e-10)
                    np.testing.assert_allclose(m['hgb'].predict(x), frame['hgb_'+arm], atol=1e-12, rtol=1e-12)
                frames.append(frame)
            print('Verified suffix readers fold', fold, flush=True)
    rows = pd.concat(frames, ignore_index=True)
    targets, summary = summarize(rows)
    roundoff = assessment_parity(targets, summary, assessment)
    result = dict(variable_folds=len(frames), readers=len(scalers)*2, scalers=len(scalers),
        patients=int(rows.patient.nunique()), query_occurrences=len(rows),
        saved_predictions_reproduced=True, original_predictions_unchanged=True,
        paired_assessment_recomputed=True, aggregate_max_roundoff=roundoff,
        post_max_error=max(m['post_max_error'] for m in parity),
        final_max_error=max(m['final_max_error'] for m in parity), checked_files=len(checked))
    output.write_text(json.dumps(dict(summary=result, runtime_modified=False, promotion_allowed=False,
        bindings={str(ap): sha256(ap), str(Path(__file__)): sha256(Path(__file__))}), indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()

"""Score frozen E23 readers; fixes aggregation without changing fitted artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.probe_organ_compartments import ARMS, CONTRASTS, DEST, KINDS, OBSERVED, assess, donor_indices, features, write_json
from scripts.train_content_retention import DEST as ENCODED
from scripts.verify_state_suffix import numeric_summary_parity
from whole_body_jepa.multisource_intervals import sha256


def joint_passes(screens, variables, controls):
    return [v for v in variables if all(v in screens[f'{r}_organs_minus_{r}_{b}']
            for r in ('ridge', 'hgb') for b in controls)]


def summarize(data, variables):
    results = {v: assess(data[data.variable == v], 20261003+i*1000) if (data.variable == v).any()
               else dict(status='insufficient_support') for i, v in enumerate(variables)}
    scored = {v: r for v, r in results.items() if 'normalized_mae' in r}
    screens = {a+'_minus_'+b: [v for v, r in scored.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
               for a, b in CONTRASTS}
    summary = dict(variables=len(scored), variable_folds=int(data[['variable', 'fold']].drop_duplicates().shape[0]),
        query_occurrences=len(data), patients=int(data.patient.nunique()),
        normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in scored.values()]))
            for a in (*ARMS, 'native')} for level in ('patient', 'component')}, screens=screens)
    summary['joint_incremental'] = joint_passes(screens, scored, ('body', 'shuffled', 'expanded'))
    summary['joint_over_raw'] = joint_passes(screens, summary['joint_incremental'], ('raw',))
    return results, summary


def verify_predictions(bundle, projection, data, rows, frame):
    expected = pd.DataFrame(dict(patient=data['patient'][rows], component=data['component'][rows],
                                 truth=data['truth'][rows], native=data['native'][rows]))
    pd.testing.assert_frame_equal(frame[expected.columns].reset_index(drop=True), expected)
    checks = {}
    for kind in KINDS:
        m = bundle[kind]
        checks[kind] = check_scaler(m['scaler'])
        x = features(data, rows, kind, projection)
        if m['hgb'].n_iter_ != 100 or m['ridge'].alpha != 100.:
            raise ValueError('Changed reader specification')
        np.testing.assert_allclose(m['ridge'].predict(m['scaler'].transform(x)),
                                   frame['ridge_'+kind], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(m['hgb'].predict(x), frame['hgb_'+kind], atol=1e-12, rtol=1e-12)
    return checks


def main():
    output = Path('artifacts/native_reconstruction_organ_compartments_assessment_20261003.json')
    verification = Path('artifacts/native_reconstruction_organ_compartments_verification_20261003.json')
    if output.exists() or verification.exists(): raise ValueError('Completed assessment already exists')
    checked, bindings, frames, scalers, parity, fit_gaps = {}, {}, [], [], [], []

    def verify(path, expected):
        path = Path(path)
        if str(path) not in checked: checked[str(path)] = sha256(path)
        if checked[str(path)] != expected: raise ValueError(f'Binding mismatch: {path}')

    with threadpool_limits(limits=4):
        for fold in range(5):
            rp = Path(f'artifacts/native_reconstruction_organ_compartments_fold{fold}_20261003.json')
            report = json.loads(rp.read_text()); bindings[str(rp)] = sha256(rp)
            contract = report['contract']; variables = contract['variables']
            assert len(variables) == 53 and set(report['targets']) == set(variables)
            parent = contract['encoded_contract']
            for group in (contract['bindings'], parent['bindings'], parent['parent']['bindings']):
                for p, h in group.items(): verify(p, h)
            dest = DEST/f'fold{fold}'; cp = dest/'contract.json'
            assert json.loads(cp.read_text()) == contract
            data = {}
            for role, meta in report['extraction'].items():
                verify(cp, meta['contract_sha256']); verify(dest/f'{role}.joblib', meta['sha256'])
                verify(OBSERVED/f'fold{fold}/{role}.joblib', meta['raw_sha256'])
                ep = ENCODED/f'fold{fold}/{role}.joblib'
                verify(ep, meta['encoded']['sha256'])
                d = joblib.load(dest/f'{role}.joblib')
                old, encoded = joblib.load(OBSERVED/f'fold{fold}/{role}.joblib'), joblib.load(ep)
                assert len(d['patient']) == len(set(d['patient'])) == len(encoded) == meta['contexts']
                np.testing.assert_array_equal(d['patient'], old['patient'])
                np.testing.assert_array_equal(d['target'], old['context'])
                np.testing.assert_array_equal(d['raw'], old['raw'])
                np.testing.assert_allclose(d['body'], old['body'], atol=1e-6, rtol=0)
                np.testing.assert_array_equal(d['donor'], donor_indices(d['patient'], d['target']))
                for i, e in enumerate(encoded):
                    assert (d['patient'][i], d['target'][i], d['component'][i], d['truth'][i], d['native'][i]) == (
                        e['patient'], e['target'], e['component'], e['truth'], e['frozen'])
                    assert not e['mask'][e['target']]
                assert np.isfinite(d['organs']).all() and d['organs'].shape[0] == len(encoded)
                assert meta['body_max_error'] <= 1e-6 and meta['native_max_error'] <= 1e-7
                parity.append(meta); data[role] = d
            f, q = data['fit'], data['selection']
            assert not set(f['patient']) & set(q['patient'])
            expected_projection = np.random.default_rng(20261003+fold).normal(size=(64, f['organs'].shape[1]))/8.
            for c, v in enumerate(variables):
                meta = report['targets'][v]
                fm, qm = np.flatnonzero(f['target'] == c), np.flatnonzero(q['target'] == c)
                assert meta['support'] == dict(fit=len(fm), selection=len(qm))
                eligible = len(fm) >= 30 and len(qm) >= 5
                assert (meta['status'] == 'fitted') == eligible
                if not eligible: continue
                verify(cp, meta['contract_sha256'])
                verify(dest/f'{v}.joblib', meta['model_sha256']); verify(dest/f'{v}.parquet', meta['prediction_sha256'])
                frame = pd.read_parquet(dest/f'{v}.parquet'); saved = joblib.load(dest/f'{v}.joblib')
                np.testing.assert_array_equal(saved['projection'], expected_projection)
                assert (frame.fold == fold).all() and (frame.variable == v).all()
                numerics = verify_predictions(saved['models'], saved['projection'], q, qm, frame)
                assert numerics == meta['numerics']; scalers.extend(numerics.values())
                fit_mae = {}
                for kind in KINDS:
                    x = features(f, fm, kind, saved['projection']); m = saved['models'][kind]
                    for reader in ('ridge', 'hgb'):
                        px = m['scaler'].transform(x) if reader == 'ridge' else x
                        fit_mae[reader+'_'+kind] = float(np.mean(np.abs(m[reader].predict(px)-f['truth'][fm])))
                numeric_summary_parity(fit_mae, meta['fit_mae'])
                fit_gaps.append(dict(fold=fold, variable=v, fit_mae=fit_mae))
                frames.append(frame)
            print('Verified compartment readers', fold, flush=True)
    data = pd.concat(frames, ignore_index=True)
    results, summary = summarize(data, variables)
    bindings[str(Path(__file__))] = sha256(Path(__file__))
    write_json(output, dict(summary=summary, targets=results, fit_diagnostics=fit_gaps, bindings=bindings,
        aggregation_repair='Compare keys include reader prefix for both candidate and control; no fitted artifacts changed.',
        runtime_modified=False, promotion_allowed=False,
        limitation='Bounded reused development masked-current inference; not forecasting, clinical validation or independent lockbox.'))
    result = dict(variable_folds=len(frames), readers=len(scalers)*2, scalers=len(scalers),
        patients=int(data.patient.nunique()), query_occurrences=len(data), checked_files=len(checked),
        saved_predictions_reproduced=True, fit_predictions_reproduced=True,
        original_body_max_error=max(m['body_max_error'] for m in parity),
        original_native_max_error=max(m['native_max_error'] for m in parity),
        organ_replay_independently_recomputed=False,
        scoring_from_saved_predictions=True, training_artifacts_modified=False)
    write_json(verification, dict(summary=result, runtime_modified=False, promotion_allowed=False,
        bindings={str(output): sha256(output), str(Path(__file__)): sha256(Path(__file__))}))
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()

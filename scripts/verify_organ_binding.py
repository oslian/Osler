"""E27 saved-model replay through the native layer API, separate from trainer loop."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_organ_binding import (
    ARMS, CONTROLS, DEST, SEED, assess, prepare, raw_control, select_arm, write_json,
)
from scripts.verify_full_organ_readout import compare_results
from whole_body_jepa.multisource_intervals import sha256


def direct_prediction(model, example, bound, arm):
    e = select_arm(example, bound, arm)
    body, organs, presence, times = [e[k] for k in ('body', 'organs', 'presence', 'times')]
    layer = model.trajectory.layer
    initial = layer.initialize(body.new_zeros((1, layer.body_dim)),
        body.new_zeros((1, layer.organ_count, layer.organ_dim)), body.new_zeros((1, layer.organ_count)))
    relative = (times-times[0]).to(body.dtype)
    state = layer.replay_observations(initial, body[None], organs[None], presence[None], relative[None])
    gap = (times.new_tensor(e['anchor'])-times[-1]).to(body.dtype).reshape(1)
    state = layer.advance(state, gap)
    return float(model.readout(state.body)[0, e['target']])


def verify_rows(frame, examples, model, bound, arm, variables):
    expected = [(e['patient'], e['component'], variables[e['target']], e['truth']) for e in examples]
    if list(frame[['patient', 'component', 'variable', 'truth']].itertuples(index=False, name=None)) != expected:
        raise ValueError('Saved prediction identities or truth changed')
    with torch.no_grad():
        prediction = np.array([direct_prediction(model, e, b, arm) for e, b in zip(examples, bound)])
    np.testing.assert_allclose(frame[arm], prediction, atol=1e-7, rtol=0)
    return float(np.max(np.abs(frame[arm].to_numpy()-prediction)))


def verify_statistics(actual, expected):
    # Parquet changes pandas block layout and thus the final mean's summation
    # order. Paired statistics, support and decisions must still match exactly.
    if ({k: v for k, v in actual.items() if k != 'normalized_mae'} !=
            {k: v for k, v in expected.items() if k != 'normalized_mae'}):
        raise ValueError('Scored statistics, support or decisions differ')
    if ('normalized_mae' in actual) != ('normalized_mae' in expected):
        raise ValueError('Changed mean-error availability')
    return compare_results(actual.get('normalized_mae'), expected.get('normalized_mae'))


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assessment_path = Path('artifacts/native_organ_binding_assessment_20261004.json')
    output = Path('artifacts/native_organ_binding_verification_20261004.json')
    if output.exists():
        raise ValueError('Verification already exists')
    assessment = json.loads(assessment_path.read_text())
    paths = [assessment_path, Path(__file__), Path('scripts/verify_full_organ_readout.py')]
    errors, frames, bindings = {}, [], {str(p): sha256(p) for p in paths}
    for fold in range(5):
        variables, examples, bound, original, contract, extraction, dest = prepare(fold)
        rp = Path(f'artifacts/native_organ_binding_fold{fold}_20261004.json')
        if sha256(rp) != assessment['bindings'][str(rp)]:
            raise ValueError('Changed fold report')
        report = json.loads(rp.read_text())
        if report['contract'] != contract or report['extraction'] != extraction:
            raise ValueError('Changed extraction or contract')
        roles = {}
        for role in examples:
            path = dest/f'{role}_predictions.parquet'
            if sha256(path) != report['prediction_hashes'][role]:
                raise ValueError('Changed saved predictions')
            roles[role] = pd.read_parquet(path)
            bindings[str(path)] = sha256(path)
        fold_errors = {}
        for arm in ARMS:
            path = dest/f'{arm}.pt'
            if sha256(path) != report['fits'][arm]['sha256']:
                raise ValueError('Changed trained weights')
            model = deepcopy(original).eval()
            model.load_state_dict(torch.load(path, weights_only=False), strict=True)
            for key, value in original.trajectory.jepa.state_dict().items():
                torch.testing.assert_close(model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
            fold_errors[arm] = {role: verify_rows(roles[role], examples[role], model, bound[role], arm, variables)
                                for role in examples}
            bindings[str(path)] = sha256(path)
        raw, raw_bindings = raw_control(fold, examples['selection'], variables)
        if raw_bindings != report['raw_bindings']:
            raise ValueError('Changed raw control binding')
        np.testing.assert_allclose(roles['selection'].raw, raw, atol=1e-12, rtol=1e-12)
        errors[str(fold)] = fold_errors
        frames.append(roles['selection']); bindings[str(rp)] = sha256(rp)
        print('E27 verified saved models', fold, fold_errors, flush=True)
    data = pd.concat(frames, ignore_index=True)
    results, roundoff = {}, 0.
    for i, variable in enumerate(variables):
        frame = data[data.variable == variable]
        expected = assess(frame, SEED+i*1000) if len(frame) else dict(status='unsupported')
        roundoff = max(roundoff, verify_statistics(expected, assessment['results'][variable]))
        results[variable] = expected
        if len(frame):
            # Independent direct patient/role grouping, without pandas loss aggregation.
            for arm in (*ARMS, 'frozen', 'raw'):
                by_patient, links = {}, {}
                for row in frame.itertuples(index=False):
                    by_patient.setdefault(row.patient, []).append(abs(getattr(row, arm)-row.truth))
                    links[row.patient] = row.component
                means = {p: sum(x)/len(x) for p, x in by_patient.items()}
                by_component = {}
                for p, value in means.items(): by_component.setdefault(links[p], []).append(value)
                independent = dict(patient=sum(means.values())/len(means),
                    component=sum(sum(x)/len(x) for x in by_component.values())/len(by_component))
                for level, value in independent.items():
                    np.testing.assert_allclose(value, expected['normalized_mae'][level][arm], atol=1e-12, rtol=1e-12)
    scored = {v: r for v, r in results.items() if r['status'] == 'scored'}
    summary = dict(patients=int(data.patient.nunique()), occurrences=len(data), targets=len(scored),
        normalized_macro={level: {arm: float(np.mean([r['normalized_mae'][level][arm] for r in scored.values()]))
            for arm in (*ARMS, 'frozen', 'raw')} for level in ('patient', 'component')},
        screens={c: [v for v, r in scored.items() if r['comparisons'][c]['screen_pass']] for c in CONTROLS})
    summary['mechanism_increment'] = sorted(set(summary['screens']['native']) & set(summary['screens']['postpool']))
    summary['full_screen'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS)))
    roundoff = max(roundoff, compare_results(summary, assessment['summary']))
    write_json(output, dict(folds=errors, trained_models=15, prediction_occurrences=len(data),
        paired_statistics_recomputed=True, independent_mae_checked=True, bindings=bindings,
        summary_roundoff_max=roundoff, summary=summary,
        runtime_modified=False, promotion_allowed=False,
        limitation='Native API replay of saved candidates; no untouched external or future-forecast validation.'))
    print('E27 full saved-model verification complete', flush=True)


if __name__ == '__main__':
    main()

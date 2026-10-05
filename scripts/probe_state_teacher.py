"""Fixed masked-raw teacher readability using the same body64 and fixed HGB."""

import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.diagnose_nonlinear_fit_gap import weighted_median
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices, verify_query
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from scripts.train_observation_skip import ROOT, PRIVATE, check_masked_target
from whole_body_jepa.multisource_intervals import sha256


SEED = 20261003
STUDENTS = ('state_student', 'process_student', 'shuffled_student')
TRUTH_REFS = ('body_probe', 'native', 'raw_probe', 'population', 'process_student', 'shuffled_student')
TEACHER_REFS = ('body_probe', 'process_student', 'shuffled_student', 'teacher_median')
ARMS = ('state_student', *TRUTH_REFS, 'teacher_median')


def process_input(raw):
    raw = np.asarray(raw, dtype=float).copy()
    if raw.ndim != 2 or raw.shape[1] != 265 or not np.isfinite(raw).all():
        raise ValueError('Invalid process input')
    raw[:, :106] = 0
    return raw


def fit_students(body, raw, teacher, people, components, seed):
    body, teacher = np.asarray(body, dtype=float), np.asarray(teacher, dtype=float)
    if (body.shape != (len(people), 64) or teacher.shape != (len(people),)
            or not np.isfinite(body).all() or not np.isfinite(teacher).all()):
        raise ValueError('Invalid student fit')
    process = process_input(raw)
    if len(process) != len(people):
        raise ValueError('Unpaired process support')
    donor = donor_indices(people, components, seed)
    weights = bounded_weights(components)
    models = {}
    with threadpool_limits(limits=4):
        for arm, x in zip(STUDENTS, (body, process, body[donor])):
            models[arm] = HistGradientBoostingRegressor(**PARAMETERS).fit(x, teacher, sample_weight=weights)
            if models[arm].n_iter_ != 100:
                raise ValueError('Incomplete student')
    return models, weighted_median(teacher, weights), int(np.count_nonzero(donor == np.arange(len(donor))))


def unit_errors(frame, truth_col):
    if (frame.empty or frame.target.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid assessment rows')
    errors = frame[list(ARMS)].subtract(frame[truth_col], axis=0).abs()
    errors['patient'] = frame.patient.to_numpy()
    people = errors.groupby('patient', sort=True)[list(ARMS)].mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    people['component'] = links.reindex(people.index)
    return people, people.groupby('component')[list(ARMS)].mean()


def assess(frame, truth_col, refs, seed, repetitions=5000):
    people, groups = unit_errors(frame, truth_col)
    comparisons = {}
    for i, ref in enumerate(refs):
        paired = dict(patient=paired_summary(people.state_student-people[ref], seed+i, repetitions),
            component=paired_summary(groups.state_student-groups[ref], seed+100+i, repetitions))
        passed = len(people) >= 30 and len(groups) >= 3 and all(
            s['mean'] < 0 and s['p_less'] is not None and s['p_less'] < .05/(47*len(refs))
            and s['ci95'] is not None and s['ci95'][1] < 0 for s in paired.values())
        comparisons[ref] = dict(paired=paired, screen=bool(passed),
                                component_mean_harms=int((groups.state_student > groups[ref]).sum()))
    return dict(query_occurrences=len(frame), unique_patients=len(people), components=len(groups),
        normalized_mae=dict(patient=people[list(ARMS)].mean().to_dict(), component=groups.mean().to_dict()),
        comparisons=comparisons, full_screen=all(c['screen'] for c in comparisons.values()),
        per_component=[dict(component=str(c), patients=int((people.component == c).sum()),
                            normalized_mae=row.to_dict()) for c, row in groups.iterrows()])


def fit_fold(fold, checked):
    dest = PRIVATE/f'state_teacher_20261003/fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('RECONSTRUCTION_STATE_TEACHER_PROTOCOL_20261003.md'),
        Path('scripts/train_nonlinear_probe.py'), Path('scripts/train_nonlinear_placebo.py'),
        Path('scripts/train_observation_skip.py'), Path('scripts/diagnose_nonlinear_fit_gap.py'),
        Path('scripts/bounded_site_sampling.py'), Path('scripts/score_additional_reconstruction.py'),
        Path('scripts/score_monitored_continuation_outer.py')]
    proof_path = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    proof = json.loads(proof_path.read_text())
    verify_bound_files(proof['bindings'], checked)
    if proof['fold'] != fold or proof['parameters'] != PARAMETERS:
        raise ValueError('Wrong parent protocol')
    paths.append(proof_path)
    paths.extend(Path(p) for p in proof['bindings'])
    roles = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds'][fold]
    variables = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)['variables']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    original_proof = json.loads(Path(f'artifacts/native_reconstruction_nonlinear_training_fold{fold}_20261001.json').read_text())
    targets = sorted(proof['targets'])
    if len(targets) != 47 or targets != sorted(original_proof['targets']):
        raise ValueError('Changed family')
    contract = dict(fold=fold, seed=SEED, parameters=PARAMETERS,
                    bindings={str(p): sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=2)+'\n')
    dbs = {role: sqlite3.connect((PRIVATE/f'fold{fold}_{role}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
           for role in ('fit', 'selection')}
    results = {}
    try:
        for target in targets:
            rp, mp, qp = dest/f'{target}.json', dest/f'{target}.joblib', dest/f'{target}.parquet'
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256'] != sha256(cp) or done['target'] != target or done['fold'] != fold
                        or done['model_sha256'] != sha256(mp) or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed complete target')
                results[target] = done
                continue
            c = variables.index(target)
            fit_ids, fitting = load_target(dbs['fit'], c)
            ids, query = load_target(dbs['selection'], c)
            groups = [components[p] for p in fit_ids]
            if any(roles[p] != 'fit' for p in fit_ids):
                raise ValueError('Invalid fitting role')
            prior = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            expected = verify_query(ids, query, prior, roles, 'selection', fit_ids, groups)
            if not np.array_equal(expected.component, [components[p] for p in ids]):
                raise ValueError('Changed query provenance')
            for values in (fitting, query):
                check_masked_target(values['raw'], c)
            bundle_path = PRIVATE/f'fold{fold}_probes/{target}.joblib'
            if sha256(bundle_path) != original_proof['targets'][target]['model_sha256']:
                raise ValueError('Changed teacher model')
            bundle = joblib.load(bundle_path)
            if bundle['parameters'] != PARAMETERS or bundle['target'] != target:
                raise ValueError('Wrong teacher contract')
            teacher, original_body = bundle['models']['raw_probe'], bundle['models']['body_probe']
            with threadpool_limits(limits=4):
                soft_fit = teacher.predict(fitting['raw'])
                soft_query = teacher.predict(query['raw'])
                body_query = original_body.predict(query['body'])
                body_fit = original_body.predict(fitting['body'])
            if (not np.allclose(soft_query, expected.raw_probe, atol=1e-12, rtol=1e-12)
                    or not np.allclose(body_query, expected.body_probe, atol=1e-12, rtol=1e-12)):
                raise ValueError('Original teacher/body parity failed')
            models, center, singletons = fit_students(fitting['body'], fitting['raw'], soft_fit,
                                                     fit_ids, groups, SEED+c)
            frame = expected.reset_index()
            frame['teacher_median'] = center
            fit_prediction = dict(body_probe=body_fit, native=fitting['prediction'],
                                  teacher_median=np.full(len(fit_ids), center))
            with threadpool_limits(limits=4):
                for arm, fx, qx in zip(STUDENTS,
                        (fitting['body'], process_input(fitting['raw']), fitting['body']),
                        (query['body'], process_input(query['raw']), query['body'])):
                    frame[arm] = models[arm].predict(qx)
                    fit_prediction[arm] = models[arm].predict(fx)
            if not np.isfinite(frame[list(ARMS)].to_numpy()).all():
                raise ValueError('Nonfinite student output')
            fit_error = pd.DataFrame({a: abs(p-soft_fit) for a, p in fit_prediction.items()})
            fit_metrics = dict(patient=fit_error.mean().to_dict(),
                component=fit_error.groupby(np.asarray(groups)).mean().mean().to_dict(),
                bounded={a: float(np.average(v, weights=bounded_weights(groups))) for a, v in fit_error.items()})
            tmp = mp.with_suffix('.tmp'); joblib.dump(models, tmp); tmp.replace(mp)
            tmp = qp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(qp)
            done = dict(fold=fold, target=target, fit_patients=len(fit_ids), selection_patients=len(ids),
                fit_imitation_mae=fit_metrics, unchanged_singletons=singletons,
                teacher_median=center, model_sha256=sha256(mp), prediction_sha256=sha256(qp),
                contract_sha256=sha256(cp))
            rp.write_text(json.dumps(done, indent=2, allow_nan=False)+'\n')
            results[target] = done
            print('Teacher readout fit', fold, target, len(results), 'of47', flush=True)
    finally:
        for db in dbs.values():
            db.close()
    report_path = Path(f'artifacts/native_reconstruction_state_teacher_fold{fold}_20261003.json')
    report = dict(contract, targets=results, runtime_modified=False, promotion_allowed=False,
        limitation='In-sample fixed teacher outputs on fitting rows; unseen selection only at assessment. Readout imitation not encoder retention proof or future capability. Fit shuffled metrics evaluate own state, not shuffled assignment training loss.')
    if report_path.exists() and json.loads(report_path.read_text()) != report:
        raise ValueError('Changed complete report')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report_path


def score(paths):
    output = Path('artifacts/native_reconstruction_state_teacher_assessment_20261003.json')
    if output.exists():
        raise ValueError('Existing complete assessment')
    frames, fits, bindings = [], {}, [Path(__file__), Path('scripts/score_additional_reconstruction.py')]
    original_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    original = json.loads(original_path.read_text())
    checked = {}
    verify_bound_files(original['bindings'], checked)
    bindings.append(original_path)
    for p in paths:
        report = json.loads(p.read_text()); f = report['fold']
        verify_bound_files(report['bindings'], checked)
        dest = PRIVATE/f'state_teacher_20261003/fold{f}'
        cp = dest/'contract.json'
        if json.loads(cp.read_text())['bindings'] != report['bindings']:
            raise ValueError('Wrong final contract')
        bindings.extend((p, cp)); fits[str(f)] = {}
        for target, meta in report['targets'].items():
            qp, mp = dest/f'{target}.parquet', dest/f'{target}.joblib'
            if (meta['contract_sha256'] != sha256(cp) or meta['prediction_sha256'] != sha256(qp)
                    or meta['model_sha256'] != sha256(mp)):
                raise ValueError('Changed final artifact')
            frame = pd.read_parquet(qp)
            old = pd.read_parquet(PRIVATE/f'fold{f}_probes/{target}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient) != set(old.patient)
                    or not (frame.fold == f).all() or not (frame.target == target).all()
                    or len(frame) != meta['selection_patients']):
                raise ValueError('Changed assessment identity')
            old = old.set_index('patient').loc[frame.patient]
            for col in ('component', 'truth', 'native', 'body_probe', 'raw_probe', 'population'):
                if not np.array_equal(frame[col].to_numpy(), old[col].to_numpy()):
                    raise ValueError('Changed reference data')
            frames.append(frame); bindings.extend((qp, mp))
            fits[str(f)][target] = meta['fit_imitation_mae']
    data = pd.concat(frames, ignore_index=True)
    if len(paths) != 5 or sorted(data.fold.unique()) != list(range(5)) or data.target.nunique() != 47:
        raise ValueError('Incomplete family')
    results = {}
    for i, target in enumerate(sorted(data.target.unique())):
        frame = data.loc[data.target == target]
        results[target] = dict(truth=assess(frame, 'truth', TRUTH_REFS, SEED+i*1000),
            teacher=assess(frame, 'raw_probe', TEACHER_REFS, SEED+i*1000+500))
        for level in ('patient', 'component'):
            for arm in ('native', 'body_probe', 'raw_probe', 'population'):
                if not np.isclose(results[target]['truth']['normalized_mae'][level][arm],
                    original['targets'][target]['normalized_mae'][level][arm], atol=1e-12, rtol=1e-12):
                    raise ValueError('Changed all-fold reference metric')
        print('Teacher readout scored', i+1, 'of47', flush=True)
    summary = dict(query_occurrences=len(data), unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient', 'target']].drop_duplicates()))
    for task, refs in (('truth', TRUTH_REFS), ('teacher', TEACHER_REFS)):
        summary[task] = dict(full_screen=[t for t, r in results.items() if r[task]['full_screen']],
            comparison_screens={ref: [t for t, r in results.items() if r[task]['comparisons'][ref]['screen']]
                                for ref in refs},
            normalized_target_macro={level: {arm: float(np.mean([r[task]['normalized_mae'][level][arm]
                for r in results.values()])) for arm in ARMS} for level in ('patient', 'component')})
    output.write_text(json.dumps(dict(summary=summary, targets=results, fit_imitation=fits,
        bindings={str(p): sha256(p) for p in bindings}, runtime_modified=False, promotion_allowed=False,
        limitation='Source-only reused masked current reconstruction; imitation of in-sample trained teacher not true-label efficacy. Two separately corrected descriptive families, conditional bootstrap, overlapping fitted models. Negative imitation not proof of lost information. No runtime, intervals or future prediction change.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    checked = {}
    score([fit_fold(f, checked) for f in range(5)])

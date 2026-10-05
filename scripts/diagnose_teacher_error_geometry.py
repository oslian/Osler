"""Exact, retrospective absolute-loss decomposition of fixed student updates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.probe_state_teacher import PRIVATE
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


POSITIONS = ('unchanged', 'teacher_stationary', 'away_teacher', 'within_teacher_segment', 'beyond_teacher')
TRUTH_POSITIONS = ('unchanged', 'away_truth', 'approaching_truth', 'crossing_truth')
QUALITIES = ('better', 'equal', 'worse')
METRICS = ('fraction', 'gross_harm', 'gross_benefit', 'net_truth_delta', 'net_imitation_delta')


def decompose(base, teacher, student, truth):
    base, teacher, student, truth = [np.asarray(v, dtype=float) for v in (base, teacher, student, truth)]
    if (base.ndim != 1 or not len(base) or any(v.shape != base.shape for v in (teacher, student, truth))
            or not all(np.isfinite(v).all() for v in (base, teacher, student, truth))):
        raise ValueError('Invalid loss geometry inputs')
    move, proposal = student-base, teacher-base
    size = abs(move)
    toward = np.sign(move) == np.sign(proposal)
    position = np.full(len(base), 'away_teacher', dtype=object)
    position[toward & (size <= abs(proposal))] = 'within_teacher_segment'
    position[toward & (size > abs(proposal))] = 'beyond_teacher'
    position[proposal == 0] = 'teacher_stationary'
    position[move == 0] = 'unchanged'
    z = (truth-base)*np.sign(move)
    truth_position = np.full(len(base), 'crossing_truth', dtype=object)
    truth_position[z <= 0] = 'away_truth'
    truth_position[z >= size] = 'approaching_truth'
    truth_position[move == 0] = 'unchanged'
    delta = abs(student-truth)-abs(base-truth)
    identity = size-2*np.clip(z, 0, size)
    if not np.allclose(identity, delta, atol=1e-10, rtol=1e-10):
        raise ValueError('Absolute-loss identity failed')
    teacher_delta = abs(teacher-truth)-abs(base-truth)
    quality = np.where(teacher_delta < 0, 'better', np.where(teacher_delta > 0, 'worse', 'equal'))
    if np.any((quality == 'better') & (position == 'within_teacher_segment') & (delta > 1e-10)):
        raise ValueError('Convex segment contradicts absolute-loss geometry')
    return pd.DataFrame(dict(teacher_position=position, truth_position=truth_position,
        teacher_quality=quality, fraction=np.ones(len(base)), gross_harm=np.maximum(delta, 0),
        gross_benefit=np.maximum(-delta, 0), net_truth_delta=delta,
        net_imitation_delta=abs(student-teacher)-abs(base-teacher)))


def aggregate(frame):
    if (frame.empty or frame.target.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1):
        raise ValueError('Invalid geometry identities')
    rows = decompose(frame.body_probe, frame.raw_probe, frame.state_student, frame.truth)
    rows['patient'] = frame.patient.to_numpy()
    rows['component'] = frame.component.to_numpy()
    links = rows.drop_duplicates('patient').set_index('patient').component

    def average(keep):
        part = rows[list(METRICS)].mul(np.asarray(keep, dtype=float), axis=0)
        part['patient'] = rows.patient
        people = part.groupby('patient', sort=True)[list(METRICS)].mean()
        groups = people.groupby(links.reindex(people.index))[list(METRICS)].mean()
        return dict(occurrences=int(np.count_nonzero(keep)),
            patients_with_any=int(rows.loc[keep, 'patient'].nunique()),
            patient=people.mean().to_dict(), component=groups.mean().to_dict())

    total = average(np.ones(len(rows), dtype=bool))
    partitions = {}
    for column, categories in (('teacher_position', POSITIONS), ('truth_position', TRUTH_POSITIONS),
                                ('teacher_quality', QUALITIES)):
        partitions[column] = {cat: average(rows[column] == cat) for cat in categories}
    joint = {q+'__'+p: average((rows.teacher_quality == q) & (rows.teacher_position == p))
             for q in QUALITIES for p in POSITIONS}
    partitions['quality_by_position'] = joint
    for categories in partitions.values():
        if sum(r['occurrences'] for r in categories.values()) != len(rows):
            raise ValueError('Partition is incomplete')
        for level in ('patient', 'component'):
            for metric in METRICS:
                if not np.isclose(sum(r[level][metric] for r in categories.values()), total[level][metric],
                                  atol=1e-12, rtol=1e-10):
                    raise ValueError('Weighted contributions do not sum')
    return dict(total=total, partitions=partitions,
        fold_occurrences={str(f): int((frame.fold == f).sum()) for f in sorted(frame.fold.unique())})


def main():
    out = Path('artifacts/native_reconstruction_teacher_error_geometry_20261003.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    parent_path = Path('artifacts/native_reconstruction_state_teacher_assessment_20261003.json')
    parent = json.loads(parent_path.read_text())
    checked = {}
    verify_bound_files(parent['bindings'], checked)
    bindings = [parent_path, Path(__file__), Path('scripts/probe_state_teacher.py'),
                Path('RECONSTRUCTION_TEACHER_ERROR_GEOMETRY_PROTOCOL_20261003.md')]
    targets = sorted(parent['targets'])
    if len(targets) != 47:
        raise ValueError('Changed family')
    results = {}
    for target in targets:
        frames = []
        for f in range(5):
            path = PRIVATE/f'state_teacher_20261003/fold{f}/{target}.parquet'
            if sha256(path) != parent['bindings'][str(path)]:
                raise ValueError('Changed private predictions')
            frame = pd.read_parquet(path)
            if not (frame.fold == f).all() or not (frame.target == target).all():
                raise ValueError('Wrong fold/target')
            frames.append(frame)
            bindings.append(path)
        full = pd.concat(frames, ignore_index=True)
        results[target] = aggregate(full)
        for level in ('patient', 'component'):
            for task, key in (('truth', 'net_truth_delta'), ('teacher', 'net_imitation_delta')):
                metrics = parent['targets'][target][task]['normalized_mae'][level]
                if not np.isclose(results[target]['total'][level][key],
                                  metrics['state_student']-metrics['body_probe'], atol=1e-12, rtol=1e-10):
                    raise ValueError('Original paired difference not reproduced')
        print('Decomposed teacher error', target, flush=True)
    summary = dict(query_occurrences=sum(r['total']['occurrences'] for r in results.values()),
        targets=len(results),
        target_macro={level: {m: float(np.mean([r['total'][level][m] for r in results.values()]))
                    for m in METRICS} for level in ('patient', 'component')},
        partition_macro={name: {cat: {level: {m: float(np.mean([
            r['partitions'][name][cat][level][m] for r in results.values()])) for m in METRICS}
            for level in ('patient', 'component')} for cat in next(iter(results.values()))['partitions'][name]}
            for name in next(iter(results.values()))['partitions']})
    if summary['query_occurrences'] != parent['summary']['query_occurrences']:
        raise ValueError('Lost original queries')
    out.write_text(json.dumps(dict(summary=summary, targets=results,
        bindings={str(p): sha256(p) for p in bindings}, runtime_modified=False, promotion_allowed=False,
        limitation='Retrospective outcome-defined descriptive geometry, NOT legal patient-state features or a routing policy. No significance testing, fitting, external evaluation, predictor clipping or deployment change. Patient-normalized fold contributions sum exactly.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary['target_macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()

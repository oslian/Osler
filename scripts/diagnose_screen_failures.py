"""E41: audit fixed E40 failure reasons without changing its acceptance policy."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from whole_body_jepa.multisource_intervals import sha256

SOURCE = Path('artifacts/native_actual_step_20261005.json')
VERIFICATION = Path('artifacts/native_actual_step_verification_20261005.json')
OUTPUT = Path('artifacts/native_screen_failures_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_SCREEN_DIAGNOSIS_PROTOCOL_20261005.md')
PRIVATE = Path('data/research_recovery/additional_development_domains_20260921/actual_step_20261005')
PRIMARY = 'actual_step'
CONTROLS = ('pcgrad', 'mean', 'fixed_immediate', 'immediate', 'fixed_delayed',
            'fixed_body', 'bound_start', 'unbridged', 'frozen', 'raw')
ARMS = (PRIMARY, *CONTROLS)
ALPHA = .05/(53*len(CONTROLS))
ORDER = ('support', 'mean', 'interval', 'nominal_test', 'family_test')
CATEGORIES = ('insufficient_support', 'mean_not_better', 'interval_inconclusive',
              'nominal_test_inconclusive', 'multiplicity_only')


def first_failure(criteria):
    return next((label for key, label in zip(ORDER, CATEGORIES) if not criteria[key]), 'passed')


def classify(entry):
    if entry['status'] == 'unsupported':
        if set(entry) != {'status'}:
            raise ValueError('Unsupported entry has unexplained fields')
        return dict(first_unmet='unsupported', comparisons={})
    if entry['status'] != 'scored' or set(entry['comparisons']) != set(CONTROLS):
        raise ValueError('Incomplete fixed comparator family')
    if any(type(entry[k]) is not int or entry[k] < 1 for k in ('patients', 'components', 'occurrences')):
        raise ValueError('Invalid support count')
    if not entry['components'] <= entry['patients'] <= entry['occurrences']:
        raise ValueError('Contradictory support counts')
    result = {}
    for control in CONTROLS:
        comparison = entry['comparisons'][control]
        stats = comparison['paired']
        if set(stats) != {'patient', 'component'}:
            raise ValueError('Missing paired level')
        for level, count in (('patient', 'patients'), ('component', 'components')):
            s = stats[level]
            if s['n'] != entry[count] or not math.isfinite(s['mean']):
                raise ValueError('Invalid paired support or mean')
            if s['n'] < 2:
                if s['p_less'] is not None or s['ci95'] is not None:
                    raise ValueError('Singleton inference must be unavailable')
            elif (s['p_less'] is None or not math.isfinite(s['p_less']) or not 0 <= s['p_less'] <= 1
                  or s['ci95'] is None or len(s['ci95']) != 2
                  or not all(math.isfinite(x) for x in s['ci95']) or s['ci95'][0] > s['ci95'][1]):
                raise ValueError('Invalid paired inference')
        flags = dict(support=entry['patients'] >= 30 and entry['components'] >= 3,
                     mean=all(s['mean'] < 0 for s in stats.values()),
                     interval=all(s['ci95'] is not None and s['ci95'][1] < 0 for s in stats.values()),
                     nominal_test=all(s['p_less'] is not None and s['p_less'] < .05 for s in stats.values()),
                     family_test=all(s['p_less'] is not None and s['p_less'] < ALPHA for s in stats.values()))
        original = flags['support'] and flags['interval'] and flags['family_test']
        if type(comparison['screen_pass']) is not bool or original != comparison['screen_pass']:
            raise ValueError('Saved gate mismatch')
        if original and not all(flags.values()):
            raise ValueError('Inference contradicts mean sign')
        result[control] = dict(criteria=flags, first_unmet=first_failure(flags),
                               all_failed=[key for key in ORDER if not flags[key]],
                               mean_difference={level: s['mean'] for level, s in stats.items()})
    joined = {key: all(r['criteria'][key] for r in result.values()) for key in ORDER}
    return dict(first_unmet=first_failure(joined), criteria=joined,
                all_failed=[key for key in ORDER if not joined[key]], comparisons=result)


def scalar_tables(frame):
    """Independent scalar aggregation; never serialize these patient-keyed tables."""
    required = ['fold', 'patient', 'variable', 'component', 'truth', *ARMS]
    if (not set(required) <= set(frame) or frame.empty or frame[required].isna().any().any()
            or frame.variable.nunique() != 1 or frame.duplicated(['fold', 'patient', 'variable']).any()
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy(float)).all()):
        raise ValueError('Invalid private paired rows')
    people, links = {}, {}
    for patient, component, truth, *predictions in frame[['patient', 'component', 'truth', *ARMS]].itertuples(index=False, name=None):
        if patient in links and links[patient] != component:
            raise ValueError('Patient assigned to multiple components')
        links[patient] = component
        people.setdefault(patient, []).append([abs(float(p)-float(truth)) for p in predictions])
    means = {p: [math.fsum(row[i] for row in rows)/len(rows) for i in range(len(ARMS))]
             for p, rows in people.items()}
    groups = {}
    for p, values in means.items():
        groups.setdefault(links[p], []).append(values)
    components = [[math.fsum(row[i] for row in rows)/len(rows) for i in range(len(ARMS))]
                  for rows in groups.values()]
    return list(means.values()), components


def scalar_paired(values):
    n = len(values)
    mean = math.fsum(values)/n
    if n < 2:
        return mean, None
    if max(values) == min(values):
        return mean, 0. if mean < 0 else 1.
    variance = math.fsum((v-mean)**2 for v in values)/(n-1)
    return mean, float(t.cdf(mean/math.sqrt(variance/n), n-1))


def verify_entry(frame, entry):
    tables = scalar_tables(frame)
    if (len(frame), len(tables[0]), len(tables[1])) != (entry['occurrences'], entry['patients'], entry['components']):
        raise ValueError('Saved support mismatch')
    maximum = 0.

    def close(actual, expected):
        nonlocal maximum
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError('Availability mismatch')
            return
        error = abs(actual-expected)
        maximum = max(maximum, error)
        if not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError('Scalar statistic mismatch')

    for level, rows in zip(('patient', 'component'), tables):
        for i, arm in enumerate(ARMS):
            close(math.fsum(row[i] for row in rows)/len(rows), entry['normalized_mae'][level][arm])
        for i, control in enumerate(CONTROLS, 1):
            mean, p = scalar_paired([row[0]-row[i] for row in rows])
            saved = entry['comparisons'][control]['paired'][level]
            close(mean, saved['mean']); close(p, saved['p_less'])
    return maximum


def summarize(results):
    counts = Counter(r['first_unmet'] for r in results.values())
    cumulative = {}
    for end, key in enumerate(ORDER, 1):
        cumulative[key] = sum(bool(r.get('criteria')) and all(r['criteria'][k] for k in ORDER[:end])
                              for r in results.values())
    control_counts = {control: dict(Counter(r['comparisons'][control]['first_unmet']
        if r['comparisons'] else 'unsupported' for r in results.values())) for control in CONTROLS}
    return dict(inventory=len(results), first_unmet_counts=dict(counts), cumulative=cumulative,
                per_control=control_counts,
                all_failed_counts={key: sum(key in r.get('all_failed', []) for r in results.values()) for key in ORDER},
                full_screen=[v for v, r in results.items() if r['first_unmet'] == 'passed'])


def main():
    if OUTPUT.exists():
        raise ValueError('Refusing to overwrite completed diagnosis')
    source, checked = json.loads(SOURCE.read_text()), json.loads(VERIFICATION.read_text())
    if not source['complete'] or not checked['complete']:
        raise ValueError('Incomplete source verification')
    bindings = {}
    for report in (source, checked):
        for name, digest in report['bindings'].items():
            if name in bindings and bindings[name] != digest:
                raise ValueError('Inconsistent original bindings')
            bindings[name] = digest
    for name, digest in bindings.items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed original binding: '+name)
    paths = [PRIVATE/f'fold{fold}'/'selection.parquet' for fold in range(5)]
    frames = []
    for fold, path in enumerate(paths):
        if str(path) not in source['bindings']:
            raise ValueError('Selection rows are not bound')
        part = pd.read_parquet(path)
        if len(part) != 1024 or not part.fold.eq(fold).all():
            raise ValueError('Incomplete selection fold')
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    entries = source['results']['selection']
    if len(entries) != 53 or not set(frame.variable) <= set(entries):
        raise ValueError('Changed complete inventory')
    results, maximum = {}, 0.
    for variable, entry in entries.items():
        part = frame[frame.variable == variable]
        if part.empty != (entry['status'] == 'unsupported'):
            raise ValueError('Support availability mismatch')
        if not part.empty:
            maximum = max(maximum, verify_entry(part, entry))
        results[variable] = classify(entry)
    summary = summarize(results)
    if summary['full_screen'] != source['summary']['full_screen']:
        raise ValueError('Full screen mismatch')
    for control in CONTROLS:
        passed = [v for v, r in results.items() if r['comparisons'] and r['comparisons'][control]['first_unmet'] == 'passed']
        if passed != source['summary']['screens'][control]:
            raise ValueError('Comparator screen mismatch')
    for path in (SOURCE, VERIFICATION, PROTOCOL, Path(__file__).relative_to(Path.cwd()),
                 Path('tests/test_screen_failures.py')):
        bindings[str(path)] = sha256(path)
    report = dict(complete=True, bindings=bindings, summary=summary, targets=results,
                  maximum_scalar_roundoff=maximum, original_alpha=ALPHA,
                  role='selection', prediction_occurrences=len(frame),
                  training_performed=False, runtime_modified=False, promotion_allowed=False,
                  limitation='Retrospective failure reporting; not new inference, root-cause repair or untouched external validation')
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

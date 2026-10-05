"""E32 same-fold/source decomposition using fixed, independently verified predictions."""
from collections import Counter
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_crossfit_group_shift import decompose
from scripts.diagnose_organ_assimilation import require_verification
from scripts.probe_assimilation_transfer import DEST as QUERY_DEST
from scripts.train_organ_assimilation import DEST as FIT_DEST, validate_identities
from scripts.train_organ_binding import write_json
from whole_body_jepa.multisource_intervals import sha256

CONTROLS = ('unbridged', 'delayed', 'body', 'frozen')
PROTOCOL = Path('RECONSTRUCTION_ASSIMILATION_COMPOSITION_PROTOCOL_20261005.md')
OUTPUT = Path('artifacts/native_assimilation_composition_20261005.json')


def contrast(fit, query, control):
    tables = {}
    roles = {}
    for name, frame in (('fit', fit), ('query', query)):
        delta = (frame.immediate-frame.truth).abs() - (frame[control]-frame.truth).abs()
        table = frame[['component']].assign(delta=delta).groupby('component').delta.agg(['mean', 'count'])
        tables[name] = table
        roles[name] = dict(patients=len(frame), components=len(table),
            mean_delta=float(delta.mean()) if len(frame) else None)
    a, b = tables['query'], tables['fit']
    common = sorted(set(a.index) & set(b.index))
    status = ('missing_role_support' if fit.empty or query.empty else
              'no_common_source' if not common else 'decomposed')
    result = dict(status=status, roles=roles, common_components=len(common))
    if status != 'decomposed':
        return result
    d = decompose(a, b)
    result.update(common_fit_patients=d['broad_common_patients'],
        common_query_patients=d['selected_common_patients'],
        common_fit_fraction=d['broad_common_fraction'],
        common_query_fraction=d['selected_common_fraction'],
        common_gap=d['common_patient_mean_gap'], within=d['within_component_term'],
        composition=d['composition_term'], equal_component_gap=d['common_equal_component_gap'],
        all_source_gap=roles['query']['mean_delta']-roles['fit']['mean_delta'],
        components=[])
    for i, c in enumerate(common):
        result['components'].append(dict(ordinal=i, fit_n=int(b.loc[c, 'count']),
            query_n=int(a.loc[c, 'count']), fit_mean=float(b.loc[c, 'mean']),
            query_mean=float(a.loc[c, 'mean'])))
    result['common_components_under30_either_role'] = sum(
        min(r['fit_n'], r['query_n']) < 30 for r in result['components'])
    result['restriction_effect'] = result['all_source_gap'] - result['common_gap']
    return result


def verify_arithmetic(result):
    if result['status'] != 'decomposed':
        return
    rows = result['components']
    nf, nq = sum(r['fit_n'] for r in rows), sum(r['query_n'] for r in rows)
    # Average the two directional standardizations, independent of midpoint helper.
    ef = fsum(r['fit_n']/nf*r['query_mean'] for r in rows)
    ff = fsum(r['fit_n']/nf*r['fit_mean'] for r in rows)
    eq = fsum(r['query_n']/nq*r['query_mean'] for r in rows)
    fq = fsum(r['query_n']/nq*r['fit_mean'] for r in rows)
    expected = dict(within=((ef-ff)+(eq-fq))/2,
        composition=((eq-ef)+(fq-ff))/2, common_gap=eq-ff,
        equal_component_gap=fsum(r['query_mean']-r['fit_mean'] for r in rows)/len(rows),
        common_fit_patients=nf, common_query_patients=nq,
        common_components_under30_either_role=sum(min(r['fit_n'], r['query_n']) < 30 for r in rows))
    for k, v in expected.items():
        if not np.isclose(result[k], v, rtol=1e-12, atol=1e-12):
            raise ValueError('Independent decomposition mismatch: '+k)


def diagnose(fit, query, inventory):
    if not inventory or len(set(inventory)) != len(inventory):
        raise ValueError('Unique inventory required')
    for frame in (fit, query):
        validate_identities(frame)
        if frame.empty or not set(frame.variable) <= set(inventory):
            raise ValueError('Invalid role targets')
        if not np.isfinite(frame[['truth', 'immediate', *CONTROLS]].to_numpy()).all():
            raise ValueError('Nonfinite predictions')
    combined = pd.concat([fit, query], ignore_index=True)
    if (combined.groupby('patient').component.nunique() > 1).any():
        raise ValueError('Changed patient source')
    if ((combined.groupby('variable').context.nunique() > 1).any()
            or (combined.groupby('context').variable.nunique() > 1).any()):
        raise ValueError('Changed target identity')
    folds = {}
    for fold in sorted(set(combined.fold)):
        a, b = fit[fit.fold == fold], query[query.fold == fold]
        if a.empty or b.empty or set(a.patient) & set(b.patient):
            raise ValueError('Missing fold role or overlapping people')
        folds[str(fold)] = {}
        for target in inventory:
            aa, bb = a[a.variable == target], b[b.variable == target]
            folds[str(fold)][target] = {}
            for control in CONTROLS:
                r = contrast(aa, bb, control)
                verify_arithmetic(r)
                folds[str(fold)][target][control] = r
    summary = {}
    for c in CONTROLS:
        all_rows = [r[c] for f in folds.values() for r in f.values()]
        rows = [r for r in all_rows if r['status'] == 'decomposed']
        summary[c] = dict(status_counts=dict(Counter(r['status'] for r in all_rows)),
            within_positive=sum(r['within'] > 0 for r in rows),
            composition_positive=sum(r['composition'] > 0 for r in rows),
            common_gap_positive=sum(r['common_gap'] > 0 for r in rows),
            within_absolute_larger=sum(abs(r['within']) > abs(r['composition']) for r in rows),
            decompositions_with_all_components_ge30=sum(r['common_components_under30_either_role'] == 0 for r in rows))
    return dict(folds=folds, summary=summary)


def main():
    if OUTPUT.exists():
        raise ValueError('Completed diagnostic exists')
    paths = [PROTOCOL, Path(__file__), Path('scripts/audit_crossfit_group_shift.py'),
        Path('scripts/diagnose_organ_assimilation.py'), Path('scripts/train_organ_assimilation.py'),
        Path('scripts/train_organ_binding.py'), Path('tests/test_assimilation_composition.py')]
    assessments = {}
    for role, stem in (('fit', 'native_organ_assimilation'), ('query', 'native_assimilation_transfer')):
        date = '20261004' if role == 'fit' else '20261005'
        ap = Path(f'artifacts/{stem}_assessment_{date}.json') if role == 'fit' else Path(f'artifacts/{stem}_{date}.json')
        vp = Path(f'artifacts/{stem}_verification_{date}.json')
        report = json.loads(ap.read_text())
        require_verification(ap, json.loads(vp.read_text()))
        assessments[role] = report
        paths.extend((ap, vp))
    if set(assessments['fit']['results']) != set(assessments['query']['results']):
        raise ValueError('Inventory differs')
    frames = {'fit': [], 'query': []}
    for fold in range(5):
        for role, stem, date, directory, filename in (
                ('fit', 'native_organ_assimilation', '20261004', FIT_DEST, 'fit_predictions.parquet'),
                ('query', 'native_assimilation_transfer', '20261005', QUERY_DEST, 'predictions.parquet')):
            rp = Path(f'artifacts/{stem}_fold{fold}_{date}.json')
            if sha256(rp) != assessments[role]['bindings'][str(rp)]:
                raise ValueError('Changed fold report')
            r = json.loads(rp.read_text()); pp = directory/f'fold{fold}'/filename
            expected = r['prediction_hashes']['fit'] if role == 'fit' else r['prediction_sha256']
            if sha256(pp) != expected:
                raise ValueError('Changed predictions')
            f = pd.read_parquet(pp)
            if set(f.fold) != {fold}:
                raise ValueError('Wrong prediction fold')
            frames[role].append(f); paths.extend((rp, pp))
    result = diagnose(*(pd.concat(frames[r], ignore_index=True) for r in ('fit', 'query')),
                      list(assessments['fit']['results']))
    result.update(bindings={str(p): sha256(p) for p in paths}, runtime_modified=False,
        promotion_allowed=False, arithmetic_verified=True,
        limitation='Fixed development means, no causal attribution or new gate; common support is restricted, '
                   'not matched physiology. Repeated folds are not independent. See E31 encoder exposure.')
    write_json(OUTPUT, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()

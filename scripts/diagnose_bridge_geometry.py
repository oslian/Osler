"""E35: exact direction/curvature and loss-objective audit, without fitting."""
from collections import defaultdict
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_trained_bridge import check_bindings
from scripts.diagnose_organ_assimilation import require_verification
from scripts.train_frozen_bridge import DEST, NEW, OUTPUT as PARENT, ROLES
from scripts.train_organ_assimilation import validate_identities
from scripts.train_organ_binding import write_json
from whole_body_jepa.multisource_intervals import sha256

OUTPUT = Path('artifacts/native_bridge_geometry_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_BRIDGE_GEOMETRY_PROTOCOL_20261005.md')
VERIFICATION = Path('artifacts/native_frozen_bridge_verification_20261005.json')


def geometry(base, prediction, truth):
    b, p, y = [np.asarray(a, dtype=float) for a in (base, prediction, truth)]
    if (b.ndim != 1 or not len(b) or p.shape != b.shape or y.shape != b.shape
            or not all(np.isfinite(a).all() for a in (b, p, y))):
        raise ValueError('Invalid geometry input')
    r, c = y-b, p-b
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError('Nonfinite residual/correction')
    ab, ap = abs(r), abs(y-p)
    g = np.where(r == 0, abs(c), -np.sign(r)*c)
    curve = np.where(r == 0, 0., 2*np.maximum(np.sign(r)*c-ab, 0.))
    change = ap-ab
    np.testing.assert_allclose(change, g+curve, atol=1e-12, rtol=1e-12)
    smooth_b = .5*np.minimum(ab, 1.)**2 + np.maximum(ab-1., 0.)
    smooth_p = .5*np.minimum(ap, 1.)**2 + np.maximum(ap-1., 0.)
    harm = np.maximum(change, 0.)
    away = (np.sign(r)*np.sign(c) <= 0)
    result = pd.DataFrame(dict(base_mae=ab, prediction_mae=ap, mae_delta=change,
        mae_right_derivative=g, crossing_curvature=curve,
        correction_size=abs(c), gross_harm=harm, gross_benefit=np.maximum(-change, 0.),
        away_harm=np.where(away, harm, 0.), crossing_harm=np.where(away, 0., harm),
        smooth_base=smooth_b, smooth_prediction=smooth_p, smooth_delta=smooth_p-smooth_b,
        smooth_right_derivative=-np.clip(r, -1., 1.)*c))
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite derived metric')
    return result


def scalar_geometry(b, p, y):
    """Independent piecewise reference, no vector geometry or sign helper."""
    r, c = y-b, p-b
    old, new = abs(y-b), abs(y-p)
    if r > 0:
        derivative = -c
        curve = 2*(c-r) if c > r else 0.
        away = c <= 0
    elif r < 0:
        derivative = c
        curve = 2*(r-c) if c < r else 0.
        away = c >= 0
    else:
        derivative, curve, away = abs(c), 0., True
    delta = new-old
    def loss(error):
        return error*error/2 if error <= 1 else error-1/2
    slope = -c if r > 1 else c if r < -1 else -r*c
    return dict(base_mae=old, prediction_mae=new, mae_delta=delta,
        mae_right_derivative=derivative, crossing_curvature=curve,
        correction_size=abs(c), gross_harm=max(delta, 0.), gross_benefit=max(-delta, 0.),
        away_harm=max(delta, 0.) if away else 0., crossing_harm=0. if away else max(delta, 0.),
        smooth_base=loss(old), smooth_prediction=loss(new), smooth_delta=loss(new)-loss(old),
        smooth_right_derivative=slope)


def summarize(frame, inventory, channel):
    validate_identities(frame)
    if (frame.empty or channel not in NEW or len(set(inventory)) != len(inventory)
            or not set(frame.variable) <= set(inventory)):
        raise ValueError('Invalid family/channel')
    # Validate numeric columns before groupby can silently skip any missing row.
    geometry(frame.bound_start, frame[channel], frame.truth)
    result, max_error = {}, 0.
    for variable in inventory:
        f = frame[frame.variable == variable]
        if f.empty:
            result[variable] = dict(status='unsupported'); continue
        rows = geometry(f.bound_start, f[channel], f.truth)
        keys = list(rows.columns)
        rows['patient'] = f.patient.to_numpy()
        people = rows.groupby('patient', sort=True)[keys].mean()
        links = f.drop_duplicates('patient').set_index('patient').component
        groups = people.groupby(links.reindex(people.index), sort=True).mean()
        metrics = dict(occurrence=rows[keys].mean().to_dict(),
                       patient=people.mean().to_dict(), component=groups.mean().to_dict())
        # Independent scalar row arithmetic and nested fsum aggregation.
        by_patient, by_component, all_rows = defaultdict(list), defaultdict(list), []
        for i, row in enumerate(f.itertuples(index=False)):
            s = scalar_geometry(float(row.bound_start), float(getattr(row, channel)), float(row.truth))
            for k in keys:
                difference = abs(s[k]-float(rows.iloc[i][k]))
                if difference > 1e-12*(1+abs(s[k])):
                    raise ValueError('Scalar row mismatch')
                max_error = max(max_error, difference)
            by_patient[row.patient].append(s); all_rows.append(s)
        def average(items):
            return {k: fsum(r[k] for r in items)/len(items) for k in keys}
        scalar_people = {pat: average(items) for pat, items in sorted(by_patient.items())}
        for pat, row in scalar_people.items():
            by_component[links.loc[pat]].append(row)
        scalar_levels = dict(occurrence=average(all_rows), patient=average(list(scalar_people.values())),
            component=average([average(items) for _, items in sorted(by_component.items())]))
        for level, values in scalar_levels.items():
            for k, value in values.items():
                error = abs(value-metrics[level][k])
                if error > 1e-12*(1+abs(value)):
                    raise ValueError('Scalar aggregation mismatch')
                max_error = max(max_error, error)
        result[variable] = dict(status='scored', occurrences=len(f), patients=len(people),
            components=len(groups), metrics=metrics)
    return result, max_error


def classify(result, level):
    rows = {k: r['metrics'][level] for k, r in result.items() if r['status'] == 'scored'}
    cases = dict(full_harm=[], no_positive_uniform_scale_gain=[], crossing_driven_harm=[],
                 ambiguous_harm=[], smooth_improves_mae_harms=[])
    for target, m in rows.items():
        tolerance = 1e-12*(1+m['base_mae']+m['prediction_mae'])
        if m['mae_delta'] <= tolerance:
            continue
        cases['full_harm'].append(target)
        g = m['mae_right_derivative']
        if g > tolerance:
            cases['no_positive_uniform_scale_gain'].append(target)
        elif g < -tolerance:
            cases['crossing_driven_harm'].append(target)
        else:
            cases['ambiguous_harm'].append(target)
        if m['smooth_delta'] < -tolerance:
            cases['smooth_improves_mae_harms'].append(target)
    keys = list(next(iter(rows.values()))) if rows else []
    return dict(supported=len(rows), counts={k: len(v) for k, v in cases.items()}, targets=cases,
        macro={k: fsum(m[k] for m in rows.values())/len(rows) for k in keys})


def main():
    if OUTPUT.exists():
        raise ValueError('Completed diagnostic exists')
    parent, verification = json.loads(PARENT.read_text()), json.loads(VERIFICATION.read_text())
    require_verification(PARENT, verification)
    check_bindings(parent['bindings'])
    inventory = list(parent['results']['fit'])
    if len(inventory) != 53 or any(set(parent['results'][r]) != set(inventory) for r in ROLES):
        raise ValueError('Changed inventory')
    paths = [PARENT, VERIFICATION, PROTOCOL, Path(__file__), Path('tests/test_bridge_geometry.py'),
        Path('scripts/train_frozen_bridge.py'), Path('scripts/train_organ_assimilation.py'),
        Path('scripts/train_organ_binding.py'), Path('scripts/diagnose_organ_assimilation.py'),
        Path('scripts/audit_trained_bridge.py')]
    frames = {r: [] for r in ROLES}
    for fold in range(5):
        mp = Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
        if sha256(mp) != parent['bindings'][str(mp)]:
            raise ValueError('Changed fold report')
        meta = json.loads(mp.read_text()); check_bindings(meta['outputs']); paths.append(mp)
        if meta['contract']['fold'] != fold:
            raise ValueError('Wrong fold metadata')
        current = {}
        for role in ROLES:
            p = DEST/f'fold{fold}'/f'{role}.parquet'
            if meta['outputs'].get(str(p)) != sha256(p):
                raise ValueError('Unbound predictions')
            f = pd.read_parquet(p); validate_identities(f)
            if len(f) != (2048 if role == 'fit' else 1024) or not (f.fold == fold).all():
                raise ValueError('Role size/fold changed')
            current[role] = set(f.patient); frames[role].append(f); paths.append(p)
        if any(current[a] & current[b] for a, b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
            raise ValueError('Role overlap')
    bindings = {str(p): sha256(p) for p in paths}
    results, summaries, maximum = {}, {}, 0.
    for role, parts in frames.items():
        f = pd.concat(parts, ignore_index=True)
        results[role], summaries[role] = {}, {}
        for channel in NEW:
            values, error = summarize(f, inventory, channel); maximum = max(maximum, error)
            for target, value in values.items():
                old = parent['results'][role][target]
                if value['status'] != old['status']:
                    raise ValueError('Support differs from E34')
                if value['status'] != 'scored': continue
                for key in ('occurrences', 'patients', 'components'):
                    if value[key] != old[key]: raise ValueError('Changed support')
                for level in ('patient', 'component'):
                    for metric, name in (('base_mae','bound_start'),('prediction_mae',channel)):
                        np.testing.assert_allclose(value['metrics'][level][metric],
                            old['normalized_mae'][level][name], atol=1e-12, rtol=1e-12)
            results[role][channel] = values
            summaries[role][channel] = {l: classify(values, l) for l in ('occurrence','patient','component')}
            print('E35 complete', role, channel, summaries[role][channel]['patient']['counts'], flush=True)
    check_bindings(bindings)
    write_json(OUTPUT, dict(bindings=bindings, inventory=inventory, results=results, summary=summaries,
        scalar_verification_complete=True, maximum_scalar_roundoff=maximum,
        runtime_modified=False, promotion_allowed=False,
        limitation='Posthoc geometry of saved current-value outputs, not a prospective router, '
        'recurrent parameter scaling, new significance screen, or future/causal/external validation.'))


if __name__ == '__main__':
    main()

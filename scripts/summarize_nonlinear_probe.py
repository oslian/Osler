"""Descriptive robustness checks of completed probes; no fitting or selection."""

import json
from pathlib import Path

import numpy as np

from whole_body_jepa.multisource_intervals import sha256


LEVELS = ('patient', 'component')


def compare(source, outer, arm):
    if not source or set(source) != set(outer):
        raise ValueError('Changed target family')
    targets = sorted(outer)
    name = arm + '_minus_native'
    both, harm, ties = [], [], []
    for target in targets:
        means = outer[target]['normalized_mae']
        differences = [means[k][arm] - means[k]['native'] for k in LEVELS]
        if not np.isfinite(differences).all():
            raise ValueError('Nonfinite target means')
        (both if all(d < 0 for d in differences) else
         harm if any(d > 0 for d in differences) else ties).append(target)
    levels = {}
    for level in LEVELS:
        native = np.array([outer[t]['normalized_mae'][level]['native'] for t in targets])
        candidate = np.array([outer[t]['normalized_mae'][level][arm] for t in targets])
        delta = candidate - native
        if not np.isfinite(np.r_[native, candidate]).all() or np.any(native < 0):
            raise ValueError('Invalid errors')
        retained = [i for i, t in enumerate(targets) if t not in ('alt', 'ast')]
        levels[level] = dict(
            native_macro=float(native.mean()), candidate_macro=float(candidate.mean()),
            relative_reduction=float(1 - candidate.mean() / native.mean()) if native.mean() > 0 else None,
            leave_one_target_out_delta=[dict(omitted=t, mean_delta=float(np.delete(delta, i).mean()))
                for i, t in enumerate(targets) if len(targets) > 1],
            excluding_alt_ast=dict(targets=len(retained),
                native_macro=float(native[retained].mean()),
                candidate_macro=float(candidate[retained].mean())) if retained else None,
            source_outer_sign_disagreements=[t for t in targets if np.sign(
                source[t]['normalized_mae'][level][arm] - source[t]['normalized_mae'][level]['native'])
                != np.sign(outer[t]['normalized_mae'][level][arm] - outer[t]['normalized_mae'][level]['native'])])
    source_screen = {t for t in targets if source[t]['comparisons'][name]['all_8_contrasts_47_screen']}
    outer_screen = {t for t in targets if outer[t]['comparisons'][name]['all_8_contrasts_47_screen']}
    groups = {}
    for t in targets:
        rows = outer[t]['per_component']
        harms = [r for r in rows if r['normalized_mae'][arm] > r['normalized_mae']['native']]
        groups[t] = dict(total=len(rows), mean_harms=len(harms),
            harms_with_at_least_30_patients=sum(r['patients'] >= 30 for r in harms),
            supported_components=sum(r['patients'] >= 30 for r in rows))
    return dict(mean_improvement_both_levels=both, mean_harm_at_either_level=harm,
        nonworse_with_tie=ties, levels=levels, component_harms=groups,
        source_and_outer_incremental_screen=sorted(source_screen & outer_screen),
        source_only_incremental_screen=sorted(source_screen - outer_screen),
        outer_only_incremental_screen=sorted(outer_screen - source_screen))


def main():
    sp = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    op = Path('artifacts/native_reconstruction_nonlinear_outer_assessment_20261002.json')
    dest = Path('artifacts/native_reconstruction_nonlinear_robustness_20261003.json')
    if dest.exists():
        raise ValueError('Existing completed diagnostic')
    source, outer = [json.loads(p.read_text()) for p in (sp, op)]
    if len(outer['targets']) != 47:
        raise ValueError('Incomplete target family')
    result = {arm: compare(source['targets'], outer['targets'], arm)
              for arm in ('body_probe', 'raw_probe')}
    result['bindings'] = {str(p): sha256(p) for p in (Path(__file__), sp, op)}
    result['runtime_modified'] = result['promotion_allowed'] = False
    result['limitation'] = ('Post-assessment descriptive sensitivity analysis, not new hypothesis tests, '
        'model selection or gates. Excluding ALT/AST is a reported sensitivity only, not changing the '
        '47-target primary evaluation. Group mean harms are not proven statistical harms. Source and '
        'outer overlap across training roles; these are not independent studies. No patient rows.')
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(dest)


if __name__ == '__main__':
    main()

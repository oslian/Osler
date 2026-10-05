"""Separate failed means from unsupported precision without changing gates."""
import json
from collections import Counter
from pathlib import Path

from whole_body_jepa.multisource_intervals import sha256


def classify(stat, alpha=.05/47):
    if stat['n'] < 2 or stat['p_less'] is None:
        return 'insufficient_statistical_support'
    if stat['mean'] >= 0:
        return 'mean_not_better'
    if stat['p_less'] >= alpha:
        return 'negative_mean_not_family_significant'
    return 'pass'


def main():
    source = Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json')
    report = json.loads(source.read_text())
    rows, categories, blockers = {}, Counter(), Counter()
    for target, result in report['targets'].items():
        base = result['base_assessment']
        checks = {f'{control}/{level}': classify(stat)
                  for control, comparison in base['comparisons'].items()
                  for level, stat in comparison.items()}
        failures = {k:v for k,v in checks.items() if v != 'pass'}
        support = base['patients'] >= 30 and base['components'] >= 3
        passed = support and not failures
        if passed != result['base_screen']:
            raise ValueError('Diagnostic disagrees with original gate')
        category = ('pass' if passed else 'support_limited' if not support else
                    'at_least_one_mean_not_better' if 'mean_not_better' in failures.values()
                    else 'all_means_better_but_uncertain')
        categories[category] += 1
        blockers.update(failures.keys())
        groups = base['per_component']
        harms = {control: sum(g[f'real_minus_{control}'] > 0 for g in groups)
                 for control in ('process', 'population')}
        gains = {}
        for level in ('patient', 'component'):
            e = result['normalized_mae'][level]
            gains[level] = dict(real_gain=e['replacement_real']-e['real'],
                               process_gain=e['replacement_process']-e['process'],
                               numeric_advantage_change=(e['process']-e['real'])-(e['replacement_process']-e['replacement_real']))
        rows[target] = dict(category=category, failed_checks=failures, checks=checks,
                            components=base['components'], component_harms=harms,
                            sampling_gain_decomposition=gains)
    # Counts of blockers overlap; they must not be added as target totals.
    summary = dict(categories=dict(categories), failed_check_counts=dict(blockers),
        mean_failures=[t for t,r in rows.items() if r['category']=='at_least_one_mean_not_better'],
        uncertain=[t for t,r in rows.items() if r['category']=='all_means_better_but_uncertain'])
    output = dict(summary=summary,targets=rows,source_sha256=sha256(source),
                  promotion_allowed=False,runtime_modified=False,
                  limitation='Descriptive decomposition, not proof of low power, causal mechanism, or absent physiological signal. No retuned thresholds.')
    Path('artifacts/native_reconstruction_coverage_failure_diagnosis_20260929.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

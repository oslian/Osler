"""Separate real/control drift and persistent component harm, without fitting."""
import json
from pathlib import Path

import numpy as np

from scripts.diagnose_coverage_component_influence import influence
from scripts.score_additional_reconstruction import FAMILY_SIZE, paired_summary
from whole_body_jepa.multisource_intervals import sha256


def compare_components(old, new, seed):
    def index(rows):
        result = {r['component']: r for r in rows}
        if len(result) != len(rows) or len(rows) < 2:
            raise ValueError('Duplicate or insufficient components')
        return result

    before, after = index(old), index(new)
    if before.keys() != after.keys():
        raise ValueError('Changed component support')
    records = []
    for name in sorted(before):
        a, b = before[name], after[name]
        if (a['patients'], a['fold']) != (b['patients'], b['fold']) or a['patients'] <= 0:
            raise ValueError('Changed patient or fold support')
        for field in ('raw_mae', 'normalized_mae'):
            values = [r[field][arm] for r in (a, b) for arm in ('real', 'process', 'population')]
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError('Invalid error')
            if a[field]['population'] != b[field]['population']:
                raise ValueError('Changed fixed population control')
        ae, be = a['raw_mae'], b['raw_mae']
        records.append(dict(component=name, patients=a['patients'], fold=a['fold'],
            real_change=be['real']-ae['real'], process_change=be['process']-ae['process'],
            old_margin=ae['real']-ae['process'], new_margin=be['real']-be['process'],
            new_real_old_process=be['real']-ae['process'],
            old_real_new_process=ae['real']-be['process'],
            old_population_margin=ae['real']-ae['population'],
            new_population_margin=be['real']-be['population']))
    fields = ('real_change', 'process_change', 'old_margin', 'new_margin',
              'new_real_old_process', 'old_real_new_process')
    means = {field: float(np.mean([r[field] for r in records])) for field in fields}
    if not np.isclose(means['new_margin']-means['old_margin'],
                      means['real_change']-means['process_change'], atol=1e-10):
        raise ValueError('Margin decomposition mismatch')
    crossed = {field: paired_summary([r[field] for r in records], seed+i)
               for i, field in enumerate(('new_real_old_process', 'old_real_new_process'))}
    harm = {}
    for control, old_key, new_key in (
        ('population', 'old_population_margin', 'new_population_margin'),
        ('process', 'old_margin', 'new_margin')):
        harm[control] = dict(
            persistent_mean_harm=sum(r[old_key] > 0 and r[new_key] > 0 for r in records),
            supported_persistent_mean_harm=sum(r['patients'] >= 30 and r[old_key] > 0
                                                and r[new_key] > 0 for r in records),
            newly_harmed=sum(r[old_key] <= 0 and r[new_key] > 0 for r in records),
            recovered=sum(r[old_key] > 0 and r[new_key] <= 0 for r in records),
            old_influence=influence(old, control), new_influence=influence(new, control))
    return dict(means=means, crossed_component_comparisons=crossed, harm=harm, groups=records)


def main():
    old_path = Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json')
    new_path = Path('artifacts/native_reconstruction_continuation_outer_20260929.json')
    output = Path('artifacts/native_reconstruction_continuation_margin_20260930.json')
    if output.exists():
        raise ValueError('Existing diagnostic')
    old, new = [json.loads(p.read_text()) for p in (old_path, new_path)]
    if old['targets'].keys() != new['targets'].keys() or len(old['targets']) != FAMILY_SIZE:
        raise ValueError('Changed target family')
    # Bind to the assessed artifacts, not merely matching filenames or aggregate sizes.
    for name, digest in new['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed assessment dependency: '+name)
    results = {}
    for i, target in enumerate(sorted(new['targets'])):
        a, b = old['targets'][target], new['targets'][target]
        aa, bb = a['base_assessment'], b['base_assessment']
        r = compare_components(aa['per_component'], bb['per_component'], 20260930+100*i)
        r.update(old_base_screen=a['base_screen'], new_base_screen=b['base_screen'])
        # This is only a crossed component check, NOT a replacement full gate.
        r['new_real_beats_old_process_component_screen'] = all((
            r['crossed_component_comparisons']['new_real_old_process']['mean'] < 0,
            r['crossed_component_comparisons']['new_real_old_process']['p_less'] < .05/FAMILY_SIZE))
        results[target] = r
    failed = [r for r in results.values() if not r['new_base_screen']]
    summary = dict(targets=len(results), failed_targets=len(failed),
        added_base_screens=new['summary']['added'],
        added_retaining_old_process_component_screen=[t for t in new['summary']['added']
            if results[t]['new_real_beats_old_process_component_screen']],
        failed_targets_with_supported_persistent_mean_harm={
            c: sum(r['harm'][c]['supported_persistent_mean_harm'] > 0 for r in failed)
            for c in ('process', 'population')},
        failed_targets_without_supported_persistent_population_mean_harm=[
            t for t,r in results.items() if not r['new_base_screen']
            and not r['harm']['population']['supported_persistent_mean_harm']])
    report = dict(summary=summary, targets=results,
        bindings={str(p):sha256(p) for p in (old_path,new_path,Path(__file__),
            Path('scripts/diagnose_coverage_component_influence.py'),
            Path('scripts/score_additional_reconstruction.py'))},
        promotion_allowed=False, runtime_modified=False,
        limitation='Posthoc paired diagnostic on reused development. Component mean harm is not statistically established harm. Overlapping fits induce dependence. Crossed component screens are not full patient/component gates. Fixed population isolates control drift algebraically, not a biological cause. No patient or component exclusions, thresholds changed, or model fitting.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

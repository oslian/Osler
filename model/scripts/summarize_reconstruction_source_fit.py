"""Keep source-fit, selection and heldout evidence separate for every target."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def main():
    seeds = (19, 31, 43, 59, 71, 83)
    output = Path('artifacts/native_reconstruction_source_fit_summary_20260917.json')
    if output.exists():
        raise ValueError('existing evidence')
    base = Path('artifacts')
    catalog_path = base / 'native_reconstruction_seven_split_summary_20260917.json'
    catalog = json.loads(catalog_path.read_text())['targets']
    bindings = {str(p): sha256(p) for p in (Path(__file__), catalog_path)}
    reports = {}
    pairs = 0
    for seed in seeds:
        paths = [base / f'native_reconstruction_source_fit_{seed}_20260917.json',
                 base / f'native_reconstruction_replication_v2_{seed}_evaluation_20260914.json']
        source, heldout = [json.loads(p.read_text()) for p in paths]
        if not source['complete'] or not source['tune_score_reproduced'] or not heldout['complete']:
            raise ValueError('incomplete evidence')
        for p, h in source['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError('source replay dependency changed')
        bindings.update({str(p): sha256(p) for p in paths})
        reports[seed] = {**source['roles'], **heldout['roles']}
        pairs += sum(v['patients'] for r in source['roles'].values() for v in r.values())
    targets = {}
    for target in catalog:
        details = {}
        for role in ('fit', 'tune', 'evaluation', 'hospital'):
            items = {}
            for seed in seeds:
                r = reports[seed][role].get(target)
                if r is None:
                    continue
                diff = r['normalized_mae']['real'] - r['normalized_mae']['process']
                items[str(seed)] = dict(patients=r['patients'], real_minus_process=diff,
                    heldout_nominal_comparison=r.get('comparisons', {}).get('process', {}).get('real'))
            details[role] = dict(seeds=items, numerical_gain_count=sum(v['real_minus_process']<0 for v in items.values()),
                available_fits=len(items), median_difference=float(np.median([v['real_minus_process'] for v in items.values()])))
        targets[target] = details
    result = dict(complete=True, seeds=list(seeds), replayed_source_pairs=pairs, targets=targets,
        bindings=bindings, runtime_modified=False, promotion_allowed=False,
        limitation='Six completed V2 replications, not the original seed7. Overlapping patients and reused hospital cohort. Fit/tune differences descriptive, tune selected checkpoints; numeric signs are not significance or causality. All catalog targets retained.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('complete', output, 'source pairs', pairs)


if __name__ == '__main__':
    main()

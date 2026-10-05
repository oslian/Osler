"""Count the collision's input support, never prediction errors or future labels."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES
from scripts.audit_organ_assignment import ROOT
from scripts.encoder_fold_scaling import transform_example
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.reconstruction_cache import ReconstructionCache

PROTOCOL = Path('RECONSTRUCTION_ORGAN_ASSIGNMENT_SUPPORT_20261004.md')
OUTPUT = Path('artifacts/native_organ_assignment_support_20261004.json')


def support(tokens, variables):
    n = len(variables)
    mask = tokens[:, n:2*n].numpy()
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected binary observation masks')
    values, ages = tokens[:, :n].numpy(), tokens[:, 2*n:3*n].numpy()
    if not np.isfinite(values[mask.astype(bool)]).all() or not np.isfinite(ages[mask.astype(bool)]).all():
        raise ValueError('Nonfinite present observation')
    result = {}
    any_multi = np.zeros(len(tokens), dtype=bool)
    any_distinct = any_multi.copy()
    for organ, members in CANONICAL_ORGAN_VARIABLES.items():
        columns = [i for i, variable in enumerate(variables) if variable in members]
        selected = mask[:, columns].astype(bool)
        multi = selected.sum(1) >= 2
        distinct = np.zeros(len(tokens), dtype=bool)
        for row in np.flatnonzero(multi):
            active = np.asarray(columns)[selected[row]]
            distinct[row] = np.ptp(values[row, active]) > 0 or np.ptp(ages[row, active]) > 0
        result[organ] = dict(multi_events=int(multi.sum()), distinct_events=int(distinct.sum()))
        any_multi |= multi
        any_distinct |= distinct
    return dict(events=len(tokens), multi_events=int(any_multi.sum()),
                distinct_events=int(any_distinct.sum()), organs=result)


def main():
    if OUTPUT.exists():
        raise ValueError('Fresh output required')
    paths = [Path(__file__), PROTOCOL, Path('whole_body_jepa/native_body_reconstruction.py'),
             Path('whole_body_jepa/reconstruction_cache.py'), Path('scripts/encoder_fold_scaling.py'),
             Path('osler_jepa/organ_schema.py')]
    fp, sp, hp = [ROOT / name for name in ('partitions_20260922.pt',
                                         'scaling_20260922.pt', 'full_histories_20260922.sqlite')]
    source_hashes = {str(p): sha256(p) for p in (fp, sp, hp)}
    plans = torch.load(fp, weights_only=False)['folds']
    scaling = torch.load(sp, weights_only=False)
    variables = scaling['variables']
    source = ReconstructionCache(hp)
    results, people = [], set()
    try:
        for fold in range(5):
            folder = ROOT / f'observed_content_20261003/fold{fold}'
            cp, dp, mp = folder/'contract.json', folder/'fit.joblib', folder/'fit.json'
            contract, meta = json.loads(cp.read_text()), json.loads(mp.read_text())
            if contract['fold'] != fold or contract['variables'] != variables:
                raise ValueError('Wrong original context scope')
            if any(contract['bindings'][p] != digest for p, digest in source_hashes.items()):
                raise ValueError('Changed source data or roles')
            if meta['sha256'] != sha256(dp) or meta['contract_sha256'] != sha256(cp):
                raise ValueError('Changed fitting context cache')
            data = joblib.load(dp)
            keys = list(zip(data['patient'], map(int, data['context'])))
            if len({p for p, _ in keys}) != len(keys) or any(plans[fold][p] != 'fit' for p, _ in keys):
                raise ValueError('Repeated or non-fitting patient')
            total = dict(fold=fold, contexts=len(keys), events=0, multi_events=0, distinct_events=0,
                         multi_contexts=0, distinct_contexts=0, organs={organ: dict(
                         multi_events=0, distinct_events=0, multi_contexts=0, distinct_contexts=0)
                         for organ in CANONICAL_ORGAN_VARIABLES})
            for patient, target in keys:
                example = transform_example(source.get(patient, target), scaling['scalers'][fold])
                clean, _ = mask_native_target(example['tokens'], example['event_hours'], variables, variables[target])
                counts = support(clean, variables)
                for name in ('events', 'multi_events', 'distinct_events'):
                    total[name] += counts[name]
                for name in ('multi', 'distinct'):
                    total[name+'_contexts'] += int(counts[name+'_events'] > 0)
                    for organ, item in counts['organs'].items():
                        total['organs'][organ][name+'_events'] += item[name+'_events']
                        total['organs'][organ][name+'_contexts'] += int(item[name+'_events'] > 0)
                people.add(patient)
            results.append(total)
            paths.extend((cp, dp, mp))
            print(json.dumps({k: v for k, v in total.items() if k != 'organs'}), flush=True)
    finally:
        source.close()
    report = dict(results=results, unique_fitting_patients=len(people),
                  context_occurrences=sum(r['contexts'] for r in results),
                  scope='All saved E19 fitting contexts; overlapping folds, not full training population',
                  bindings={**source_hashes, **{str(p): sha256(p) for p in paths}},
                  accuracy_evaluated=False, runtime_modified=False, promotion_allowed=False)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

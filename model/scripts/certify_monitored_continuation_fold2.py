"""Certify that fold 0's continued fit is also fold 2's fit, without refitting."""

import json
from pathlib import Path

import numpy as np
import torch

from whole_body_jepa.multisource_intervals import sha256


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right)
        )
    return left == right


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    folder = root / 'monitored_continuation_20260929'
    partitions = root / 'partitions_20260922.pt'
    scaling = root / 'scaling_20260922.pt'
    prior = Path('artifacts/native_reconstruction_coverage_fold_reuse_20260925.json')
    certificate = json.loads(prior.read_text())
    for name, digest in certificate['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed original reuse dependency: ' + name)
    for pair in (certificate['sources'], certificate['destinations']):
        for entry in pair.values():
            if sha256(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Changed epoch-8 checkpoint')
    roles = torch.load(partitions, weights_only=False)['folds']
    fit0 = {patient for patient, role in roles[0].items() if role == 'fit'}
    fit2 = {patient for patient, role in roles[2].items() if role == 'fit'}
    if fit0 != fit2:
        raise ValueError('Fold 0 and 2 fit patients differ')
    scalers = torch.load(scaling, weights_only=False)['scalers']
    if not same(scalers[0], scalers[2]):
        raise ValueError('Fold 0 and 2 scaling differs')

    paths = [Path(__file__), prior, partitions, scaling]
    arms = {}
    for arm in ('real', 'process'):
        parent0 = Path(certificate['sources'][arm]['path'])
        parent2 = Path(certificate['destinations'][arm]['path'])
        base0 = torch.load(parent0, weights_only=False)
        base2 = torch.load(parent2, weights_only=False)
        for field in ('state_dict', 'optimizer', 'rng', 'torch_rng', 'pending',
                      'seen', 'medians', 'variables', 'targets'):
            if not same(base0[field], base2[field]):
                raise ValueError(f'Unequal epoch-8 {arm} {field}')
        epochs = {}
        for epoch in (9, 10):
            path = folder / f'fold0_{arm}_epoch{epoch}.pt'
            monitor = folder / f'fold0_{arm}_epoch{epoch}_monitor.json'
            ck = torch.load(path, weights_only=False)
            if ck['epoch'] != epoch or ck['fold'] != 0 or ck['arm'] != arm:
                raise ValueError('Unexpected continuation stage')
            if ck['continuation_bindings'][str(parent0)] != sha256(parent0):
                raise ValueError('Continuation parent mismatch')
            for name, digest in ck['continuation_bindings'].items():
                if sha256(Path(name)) != digest:
                    raise ValueError('Changed continuation dependency: ' + name)
            report = json.loads(monitor.read_text())
            if report['epoch'] != epoch or report['bindings'] != ck['continuation_bindings']:
                raise ValueError('Monitor/continuation mismatch')
            if len(report['targets']) != len(ck['targets']):
                raise ValueError('Incomplete fixed monitor')
            paths.extend((path, monitor))
            epochs[str(epoch)] = dict(checkpoint_sha256=sha256(path),
                                     monitor_sha256=sha256(monitor),
                                     fit_updates=ck['log'][-1]['updates'],
                                     unique_fit_pairs_seen=len(ck['seen']))
        paths.extend((parent0, parent2))
        arms[arm] = epochs
    result = dict(source_fold=0, destination_fold=2,
                  fit_identity_equal=True, scalers_equal=True,
                  continuing_from_equivalent_epoch8=True,
                  deterministic_training_inputs_equal=True,
                  arms=arms, bindings={str(path): sha256(path) for path in paths},
                  independently_trained=False, destination_monitor_evaluated=False,
                  promotion_allowed=False, runtime_modified=False,
                  limitation='Certifies identical fitted computation. Fold 2 has different selection queries, so its fixed monitor still requires separate inference.')
    output = Path('artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json')
    if output.exists():
        raise ValueError('Existing certificate')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'bindings'}, indent=2))


if __name__ == '__main__':
    main()

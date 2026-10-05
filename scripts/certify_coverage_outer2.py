"""Reuse identical frozen fold0 selection inference for fold2 test role."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    src = root / 'coverage_selection_20260926/fold0.parquet'
    dest = root / 'coverage_outer_20260927/fold2.parquet'
    if dest.exists() or dest.with_suffix('.json').exists():
        raise ValueError('Existing destination')
    source_meta = json.loads(src.with_suffix('.json').read_text())
    if sha256(src) != source_meta['prediction_sha256']:
        raise ValueError('Changed source predictions')
    paths = [Path(__file__), src, src.with_suffix('.json')]
    # Preserve the original inference-code digest as provenance. That script
    # has since gained an explicit role argument; trained dependencies cannot change.
    for name, digest in source_meta['bindings'].items():
        if Path(name).resolve() != Path('scripts/infer_coverage_selection.py').resolve() and sha256(Path(name)) != digest:
            raise ValueError('Changed source dependency: ' + name)
    fp = root / 'partitions_20260922.pt'
    sp = root / 'scaling_20260922.pt'
    paths.extend((fp, sp))
    folds = torch.load(fp, weights_only=False)['folds']
    if {p for p, role in folds[0].items() if role == 'selection'} != {p for p, role in folds[2].items() if role == 'test'}:
        raise ValueError('Different query patients')
    scaling = torch.load(sp, weights_only=False)
    for key in scaling['scalers'][0]:
        np.testing.assert_array_equal(scaling['scalers'][0][key], scaling['scalers'][2][key])
    for directory in ('coverage_training_20260925', 'training_20260923'):
        for arm in ('real', 'process'):
            pair = [root / directory / f'fold{i}_{arm}.pt' for i in (0, 2)]
            paths.extend(pair)
            a, b = [torch.load(p, weights_only=False) for p in pair]
            if a['epoch'] != 8 or b['epoch'] != 8 or a['medians'] != b['medians']:
                raise ValueError('Different fitted stage or population baseline')
            if a['state_dict'].keys() != b['state_dict'].keys() or not all(torch.equal(v, b['state_dict'][k]) for k, v in a['state_dict'].items()):
                raise ValueError('Different model weights')
            for ck in (a, b):
                for name, digest in ck['bindings'].items():
                    if sha256(Path(name)) != digest:
                        raise ValueError('Changed trained dependency')
    cache = root / 'full_histories_20260922.sqlite'
    report = json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    if sha256(cache) != report['private_sha256']:
        raise ValueError('Changed histories')
    frame = pd.read_parquet(src)
    old_path = root / 'outer_20260924/fold2.parquet'
    paths.append(old_path)
    old = pd.read_parquet(old_path)
    cols = ['patient', 'target', 'component', 'truth', 'fit_scale', 'population']
    pd.testing.assert_frame_equal(frame[cols], old[cols])
    np.testing.assert_array_equal(frame.replacement_real, old.real)
    np.testing.assert_array_equal(frame.replacement_process, old.process)
    if frame.duplicated(['patient', 'target']).any() or not np.isfinite(frame[['real', 'process', 'replacement_real', 'replacement_process']]).all().all():
        raise ValueError('Invalid predictions')
    frame['fold'] = 2
    frame.to_parquet(dest, index=False)
    result = dict(fold=2, role='test', examples=len(frame), patients=int(frame.patient.nunique()),
                  prediction_sha256=sha256(dest), bindings={str(p): sha256(p) for p in paths},
                  source_inference_bindings=source_meta['bindings'], cache_sha256=sha256(cache),
                  equivalent_inference_reuse=True, independently_trained=False,
                  runtime_modified=False, promotion_allowed=False,
                  limitation='Identical weights, scalers, cache and query patients; only fold/role metadata changes. Not independent replication.')
    dest.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('bindings', 'source_inference_bindings')}, indent=2))


if __name__ == '__main__':
    main()

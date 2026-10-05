"""Frozen epoch-10 predictions on the original outer patient-target grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import inverse_value, transform_example
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=range(5), type=int, required=True)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    source_fold = 0 if args.fold == 2 else args.fold
    folder = root / 'monitored_continuation_20260929'
    output_dir = root / 'continuation_outer_20260929'
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f'fold{args.fold}.parquet'
    meta = output.with_suffix('.json')
    if output.exists() or meta.exists():
        raise ValueError('Existing outer predictions')
    certificate = Path('artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json')
    if args.fold == 2:
        proof = json.loads(certificate.read_text())
        for name, digest in proof['bindings'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Changed reuse dependency: ' + name)
    fp = root / 'partitions_20260922.pt'
    sp = root / 'scaling_20260922.pt'
    cp = Path('models/whole_body_joint_jepa.pt')
    original_path = root / f'coverage_outer_20260927/fold{args.fold}.parquet'
    original_meta = json.loads(original_path.with_suffix('.json').read_text())
    if sha256(original_path) != original_meta['prediction_sha256'] or original_meta['role'] != 'test':
        raise ValueError('Changed outer baseline')
    original = pd.read_parquet(original_path)
    if original.duplicated(['patient', 'target']).any() or original.fold.nunique() != 1 or original.fold.iloc[0] != args.fold:
        raise ValueError('Invalid outer baseline keys')
    plan = torch.load(fp, weights_only=False)
    roles = plan['folds'][args.fold]
    scaling = torch.load(sp, weights_only=False)
    scaler = scaling['scalers'][args.fold]
    variables = scaling['variables']
    schema = torch.load(cp, weights_only=False)
    if schema['variables'] != variables:
        raise ValueError('Changed input schema')
    checkpoint_paths = {
        arm: folder / f'fold{source_fold}_{arm}_epoch10.pt'
        for arm in ('real', 'process')
    }
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    models = {}
    for arm, path in checkpoint_paths.items():
        ck = torch.load(path, weights_only=False)
        if ck['epoch'] != 10 or ck['fold'] != source_fold or ck['arm'] != arm:
            raise ValueError('Wrong continued checkpoint')
        for name, digest in ck['continuation_bindings'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Changed trained dependency: ' + name)
        if args.fold == 2 and sha256(path) != proof['arms'][arm]['10']['checkpoint_sha256']:
            raise ValueError('Wrong reused checkpoint')
        model = NativeBodyReconstruction(NativeOrganStateResidual(
            fresh_encoder(schema), variables, ('map',)
        )).eval()
        model.load_state_dict(ck['state_dict'], strict=True)
        models[arm] = model
    cache_path = root / 'full_histories_20260922.sqlite'
    cache_report = json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    if sha256(cache_path) != cache_report['private_sha256']:
        raise ValueError('Changed histories')
    cache = ReconstructionCache(cache_path)
    rows = []
    try:
        keys = [(patient, int(c)) for patient, c in cache.db.execute(
            'SELECT patient,target FROM examples ORDER BY patient,target'
        ) if roles[patient] == 'test']
        expected = list(zip(original.patient, original.target))
        if [(p, variables[c]) for p, c in keys] != expected:
            raise ValueError('Outer query grid changed')
        with torch.inference_mode():
            for number, (patient, c) in enumerate(keys, 1):
                example = transform_example(cache.get(patient, c), scaler)
                target = variables[c]
                predictions = {}
                for arm, model in models.items():
                    x = example['tokens']
                    if arm == 'process':
                        x = x.clone()
                        x[:, :len(variables)] = 0
                    value = float(model(x, example['event_hours'], example['anchor_hour'], target)[0])
                    predictions[arm + '10'] = float(inverse_value(value, c, scaler))
                if not all(np.isfinite(value) for value in predictions.values()):
                    raise ValueError('Nonfinite prediction')
                rows.append(dict(patient=patient, target=target, fold=args.fold, **predictions))
                if number % 10000 == 0:
                    print('Outer progress', args.fold, number, len(keys), flush=True)
    finally:
        cache.close()
    frame = pd.DataFrame(rows)
    if len(frame) != len(original) or frame.duplicated(['patient', 'target']).any():
        raise ValueError('Incomplete outer inference')
    temp = output.with_suffix('.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(output)
    paths = [Path(__file__), fp, sp, cp, original_path, cache_path,
             *checkpoint_paths.values()]
    if args.fold == 2:
        paths.append(certificate)
    report = dict(fold=args.fold, source_fold=source_fold, role='test',
                  examples=len(frame), patients=int(frame.patient.nunique()),
                  prediction_sha256=sha256(output),
                  bindings={str(path): sha256(path) for path in paths},
                  candidate_only=True, promotion_allowed=False,
                  limitation='Frozen continued model predictions on previously inspected outer development data.')
    meta.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'bindings'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Evaluate the certified fold-0 continuation on fold-2 selection patients."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import inverse_value, transform_example
from scripts.fixed_reconstruction_monitor import monitor_scores
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('real', 'process'), required=True)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    folder = root / 'monitored_continuation_20260929'
    certificate = Path('artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json')
    proof = json.loads(certificate.read_text())
    for name, digest in proof['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed reuse dependency: ' + name)
    keys_path = root / 'fixed_monitor_20260929/fold2_keys.parquet'
    baseline_path = Path('artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json')
    baseline = json.loads(baseline_path.read_text())
    if sha256(keys_path) != baseline['bindings'][str(keys_path)]:
        raise ValueError('Changed monitor keys')
    keys = pd.read_parquet(keys_path)
    scaling = torch.load(root / 'scaling_20260922.pt', weights_only=False)
    scaler = scaling['scalers'][2]
    variables = scaling['variables']
    roles = torch.load(root / 'partitions_20260922.pt', weights_only=False)['folds'][2]
    if keys.duplicated(['patient', 'target']).any() or not all(
        roles[patient] == 'selection' for patient in keys.patient
    ):
        raise ValueError('Invalid fold-2 selection identities')
    schema = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cache_path = root / 'full_histories_20260922.sqlite'
    cache_report = json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    if sha256(cache_path) != cache_report['private_sha256']:
        raise ValueError('Changed histories')
    cache = ReconstructionCache(cache_path)
    try:
        for epoch in (8, 9, 10):
            if epoch == 8:
                checkpoint_path = root / f'coverage_training_20260925/fold2_{args.arm}.pt'
            else:
                checkpoint_path = folder / f'fold0_{args.arm}_epoch{epoch}.pt'
            output = folder / f'fold2_{args.arm}_epoch{epoch}_monitor.json'
            if output.exists():
                raise ValueError('Existing fold-2 monitor')
            if epoch > 8 and sha256(checkpoint_path) != proof['arms'][args.arm][str(epoch)]['checkpoint_sha256']:
                raise ValueError('Changed continued checkpoint')
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if checkpoint['epoch'] != epoch:
                raise ValueError('Wrong model epoch')
            model = NativeBodyReconstruction(NativeOrganStateResidual(
                fresh_encoder(schema), variables, ('map',)
            )).eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            rows = []
            with torch.inference_mode():
                for row in keys.itertuples():
                    c = variables.index(row.target)
                    example = transform_example(cache.get(row.patient, c), scaler)
                    tokens = example['tokens']
                    if args.arm == 'process':
                        tokens = tokens.clone()
                        tokens[:, :len(variables)] = 0
                    value = float(model(tokens, example['event_hours'],
                                        example['anchor_hour'], row.target)[0])
                    rows.append(dict(patient=row.patient, target=row.target,
                                     component=row.component,
                                     prediction=float(inverse_value(value, c, scaler)),
                                     truth=float(inverse_value(example['truth_normalized'], c, scaler)),
                                     fit_scale=float(scaler['scale'][c])))
            scores = monitor_scores(keys, pd.DataFrame(rows))
            if len(scores) != 47:
                raise ValueError('Incomplete target family')
            if epoch == 8:
                old = baseline['folds']['2'][args.arm]
                for target, metrics in scores.items():
                    for metric in ('normalized_patient_mae', 'normalized_component_mae', 'smooth_l1'):
                        if not np.isclose(metrics[metric], old[target][metric], rtol=1e-10, atol=1e-10):
                            raise ValueError('Fold-2 epoch-8 parity failure')
            bindings = {str(path): sha256(path) for path in (
                Path(__file__), certificate, keys_path, baseline_path,
                checkpoint_path, cache_path,
            )}
            output.write_text(json.dumps(dict(fold=2, arm=args.arm, epoch=epoch,
                                               bindings=bindings, targets=scores,
                                               independently_trained=False,
                                               promotion_allowed=False), indent=2) + '\n')
            print('Fold2 monitor', args.arm, epoch, len(rows), flush=True)
    finally:
        cache.close()


if __name__ == '__main__':
    main()

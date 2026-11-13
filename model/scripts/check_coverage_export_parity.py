"""Real-history export parity smoke test; never estimates predictive quality."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.load_coverage_candidate import load
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.encoder_fold_scaling import transform_example
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--export', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    torch.set_num_threads(4)
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    schema = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
    private = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), original['variables'], ('map',))).eval()
    private.load_state_dict(original['state_dict'], strict=True)
    exported, payload = load(args.export)
    if payload['epoch'] != original['epoch']:
        raise ValueError('Export epoch mismatch')
    for key, value in original['state_dict'].items():
        if not torch.equal(value.cpu(), payload['state_dict'][key]):
            raise ValueError('Export weight mismatch')
    scaler = torch.load(root/'scaling_20260922.pt', weights_only=False)['scalers'][0]
    public_scaler = {k: v.numpy() for k, v in payload['scaler'].items()}
    for key in scaler:
        np.testing.assert_array_equal(scaler[key], public_scaler[key])
    roles = torch.load(root/'partitions_20260922.pt', weights_only=False)['folds'][0]
    cache = ReconstructionCache(root/'full_histories_20260922.sqlite')
    checked = set()
    try:
        with torch.inference_mode():
            for patient, col in cache.db.execute('SELECT patient,target FROM examples ORDER BY patient,target'):
                if roles[patient] != 'selection' or col in checked:
                    continue
                raw = cache.get(patient, col)
                a, b = transform_example(raw, scaler), transform_example(raw, public_scaler)
                target = original['variables'][col]
                y = private(a['tokens'], a['event_hours'], a['anchor_hour'], target)
                z = exported(b['tokens'], b['event_hours'], b['anchor_hour'], target)
                if not torch.equal(y, z) or not torch.isfinite(y).all():
                    raise ValueError('Export prediction mismatch')
                checked.add(col)
    finally:
        cache.close()
    result = dict(checked_targets=[original['variables'][c] for c in sorted(checked)],
                  exact_prediction_parity=True, scaler_parity=True,
                  limitation='One source-selection history per supported target, deterministic first key. No errors/labels scored; not accuracy or full evaluation.', promotion_allowed=False)
    if len(checked) != len(original['targets']):
        raise ValueError('Incomplete target parity check')
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print('Exact parity:', len(checked), 'targets')


if __name__ == '__main__':
    main()

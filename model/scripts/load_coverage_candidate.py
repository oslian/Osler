"""Strict load of sanitized research export; no private cohort is required."""
import argparse
import torch
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['format'] != 'masked-current-reconstruction-coverage-v1':
        raise ValueError('Wrong checkpoint format')
    model = NativeBodyReconstruction(NativeOrganStateResidual(
        fresh_encoder(checkpoint['schema']), checkpoint['variables'], ('map',)))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    model, checkpoint = load(args.checkpoint)
    print('Strict load OK:', len(checkpoint['targets']), 'reconstruction targets; candidate_only')

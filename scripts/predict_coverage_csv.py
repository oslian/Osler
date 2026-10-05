"""Run the research candidate from explicit canonical measurement-event CSV."""

import argparse
import json

import torch

from scripts.load_coverage_candidate import load
from whole_body_jepa.contracts import VARIABLE_UNITS
from whole_body_jepa.coverage_input import build_input, read_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--schema', action='store_true')
    parser.add_argument('--events')
    parser.add_argument('--stay-id')
    parser.add_argument('--anchor')
    parser.add_argument('--target')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load(args.checkpoint)
    if args.schema:
        print(json.dumps([dict(variable=v, unit=VARIABLE_UNITS[v],
                              reconstruction_target=v in checkpoint['targets'])
                          for v in checkpoint['variables']], indent=2))
        return
    if any(value is None for value in (args.events, args.stay_id, args.anchor, args.target)):
        parser.error('--events, --stay-id, --anchor and --target are required unless --schema')
    example = build_input(read_events(args.events), checkpoint, args.stay_id, args.anchor, args.target)
    with torch.inference_mode():
        normalized = float(model(example['tokens'], example['event_hours'], example['anchor_hour'], args.target)[0])
    c = checkpoint['variables'].index(args.target)
    point = normalized * float(checkpoint['scaler']['scale'][c]) + float(checkpoint['scaler']['location'][c])
    if not torch.isfinite(torch.tensor(point)):
        raise ValueError('Nonfinite model estimate')
    print(json.dumps(dict(task='masked_current_reconstruction', target=args.target,
        current_estimate=point, unit=VARIABLE_UNITS[args.target], anchor=args.anchor,
        independent_event_groups=len(example['tokens']),
        newest_independent_event_age_hours=example['anchor_hour']-float(example['event_hours'][-1]),
        source_fold=checkpoint['fold'], lower=None, upper=None, interval_status='not_calibrated',
        candidate_only=True, causal_claim_allowed=False,
        warning='Research candidate, not future forecast or clinical decision. Synthetic demo values do not validate accuracy.'), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

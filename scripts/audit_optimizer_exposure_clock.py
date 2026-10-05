"""Measure optimizer memory/decay per example, without refitting any model."""
import json
import math
from pathlib import Path
import torch
from collections import Counter
from whole_body_jepa.multisource_intervals import sha256


def matched_decay(reference_lr, reference_decay, examples_per_step, new_lr):
    if not 0 <= reference_lr * reference_decay < 1 or examples_per_step < 1 or new_lr <= 0:
        raise ValueError('Invalid decay clock')
    return -math.expm1(examples_per_step * math.log1p(-reference_lr * reference_decay)) / new_lr


def describe(optimizer, batch, expected_steps):
    config = optimizer['param_groups'][0]
    if any((g['lr'], g['betas'], g['weight_decay']) !=
           (config['lr'], config['betas'], config['weight_decay']) for g in optimizer['param_groups']):
        raise ValueError('Mixed optimizer groups')
    counts = Counter(int(s['step']) for s in optimizer['state'].values() if 'step' in s)
    steps = sorted(counts)
    if not steps or max(steps) != expected_steps or min(steps) <= 0:
        raise ValueError('Unexpected parameter step count')
    return dict(lr=config['lr'], betas=list(config['betas']), weight_decay=config['weight_decay'],
        actual_steps=steps, tensors_by_step=dict(counts), examples_per_step=batch,
        tensors_without_state=sum(len(g['params']) for g in optimizer['param_groups']) - len(optimizer['state']),
        exponential_half_life_examples=[batch * math.log(.5) / math.log(b) for b in config['betas']],
        decay_only_factor=math.exp(expected_steps * math.log1p(-config['lr'] * config['weight_decay'])),
        decay_factor_by_step={n:math.exp(n*math.log1p(-config['lr']*config['weight_decay'])) for n in steps})


def main():
    out = Path('artifacts/native_reconstruction_optimizer_clock_20260919.json')
    if out.exists():
        raise ValueError('Existing clock audit')
    root = Path('data/research_recovery/broad_source_20260917')
    files = {'sequential': root/'source_exposure_pair/expanded/real.pt',
             'same_lr': root/'accumulated_training/same_lr.pt',
             'sqrt_lr': root/'accumulated_training/sqrt_lr.pt'}
    results = {}
    for arm, path in files.items():
        ck = torch.load(path, weights_only=False)
        if ck['epoch'] != 8:
            raise ValueError('Incomplete training')
        for p, h in ck['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError('Checkpoint binding changed')
        results[arm] = describe(ck['optimizer'], 1 if arm == 'sequential' else 528,
                               283008 if arm == 'sequential' else 536)
    ref = results['sequential']
    for arm in ('same_lr', 'sqrt_lr'):
        results[arm]['decay_to_match_reference_exposure'] = matched_decay(ref['lr'], ref['weight_decay'], 528, results[arm]['lr'])
        results[arm]['betas_to_match_reference_sample_half_life'] = [b ** 528 for b in ref['betas']]
    out.write_text(json.dumps(dict(arms=results, bindings={str(p):sha256(p) for p in [Path(__file__), *files.values()]},
        runtime_modified=False, limitation='Clock and decay arithmetic only. Decay-only factor excludes gradients; some parameters lack optimizer state. Matching betas does not equate moments of averaged gradients to sequential Adam.'), indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

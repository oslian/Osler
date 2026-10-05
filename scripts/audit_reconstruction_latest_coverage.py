"""Separate suffix coverage loss from removal of repeated observation history."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def latest_coverage(tokens, times, variable_count, suffix_count):
    if (suffix_count < 1 or variable_count < 1 or tokens.ndim != 2
            or tokens.shape[1] < 3 * variable_count
            or times.shape != (len(tokens),) or not len(tokens)):
        raise ValueError('Invalid observation stream')
    mask = tokens[:, variable_count:2 * variable_count].bool()
    keep = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    keep[-suffix_count:] = True
    for col in range(variable_count):
        observed = mask[:, col].nonzero().flatten()
        if len(observed):
            keep[observed[-1]] = True
    # Preserve entire simultaneous event groups, never retime old measurements.
    return tokens[keep], times[keep]


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    training_path = Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    previous_path = Path('artifacts/native_reconstruction_history_suffix_20260920.json')
    checkpoint_path = root / 'bounded_site_training/bounded.pt'
    out = Path('artifacts/native_reconstruction_latest_coverage_20260920.json')
    private = root / 'latest_coverage_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing audit')
    training, previous = [json.loads(p.read_text()) for p in (training_path, previous_path)]
    for report in (training, previous):
        for p, expected in report['bindings'].items():
            if sha256(Path(p)) != expected:
                raise ValueError('Dependency changed')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if checkpoint['bindings'] != training['bindings']:
        raise ValueError('Checkpoint mismatch')
    paths = (root / 'joint_source_manifest.pt', root / 'full_samples/index.pt',
             Path('data/research_recovery/reconstruction_nested_selection_20260917.pt'))
    joint, full, split = [torch.load(p, weights_only=False) for p in paths]
    variables, targets = joint['variables'], joint['targets']
    cols = [variables.index(t) for t in targets]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(checkpoint['best_state_dict'])
    output, raw = {}, {}
    with torch.inference_mode():
        for group in ('selected', 'broad'):
            store = SourceStore([s for s in shards if s['group'] == group and s['role'] == 'encoder_tune'], cols)
            rows = []
            for key, truth in store.labels.items():
                example = store[key]
                target = variables[key[1]]
                tokens, times = mask_native_target(example['tokens'], example['event_hours'], variables, target)
                variants = {'full': (tokens, times), 'last1': (tokens[-1:], times[-1:]),
                            'latest_each': latest_coverage(tokens, times, len(variables), 1),
                            'last8_latest_each': latest_coverage(tokens, times, len(variables), 8)}
                errors = {name: abs(float(model(x, t, example['anchor_hour'], target)[0]) - truth)
                          for name, (x, t) in variants.items()}
                rows.append(dict(patient=key[0], target=target, errors=errors,
                                 events={name: len(x) for name, (x, _) in variants.items()}))
            per = {}
            for target in targets:
                chosen = [r for r in rows if r['target'] == target]
                mae = {name: float(np.mean([r['errors'][name] for r in chosen])) for name in variants}
                for name in ('full', 'last1'):
                    if not np.isclose(mae[name], previous['groups'][group][target]['mae'][name], atol=1e-7, rtol=0):
                        raise ValueError('Previous score mismatch')
                changes = {name: interval([r['errors'][name] - r['errors']['full'] for r in chosen])
                           for name in ('latest_each', 'last8_latest_each')}
                per[target] = dict(patients=len(chosen), mae=mae, minus_full=changes,
                                   mean_events={name: float(np.mean([r['events'][name] for r in chosen])) for name in variants})
            output[group], raw[group] = per, rows
            print(group, 'complete', len(rows), flush=True)
    torch.save(raw, private)
    out.write_text(json.dumps(dict(groups=output,
        bindings={str(p): sha256(p) for p in (Path(__file__), training_path, previous_path, checkpoint_path, *paths)},
        private_sha256=sha256(private), runtime_modified=False, promotion_allowed=False,
        limitation='Original source-selection patients, fixed weights, unadjusted descriptive intervals. '
        'Removing events changes initialization and input distribution. Latest-event retention also retains '
        'co-measured variables; not an isolated causal memory intervention or a deployment rule.'), indent=2) + '\n')
    for group, per in output.items():
        print(group, {name: float(np.mean([r['mae'][name] for r in per.values()])) for name in variants})


if __name__ == '__main__':
    main()

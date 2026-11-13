"""Source-only nested batch-size reliability of trained shared update gradients."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else None


def split_prefixes(size, rng, maximum=64):
    count = min(size // 2, maximum)
    order = rng.permutation(size)
    return order[:count], order[count:2 * count]


def main():
    out = Path('artifacts/native_reconstruction_gradient_sample_size_20260919.json')
    if out.exists():
        raise ValueError('Existing audit')
    root = Path('data/research_recovery/broad_source_20260917')
    tp = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    training = json.loads(tp.read_text())
    for p, h in training['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Training changed')
    ip, ep = root / 'joint_source_manifest.pt', root / 'full_samples/index.pt'
    sp = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint, full, split = [torch.load(p, weights_only=False) for p in (ip, ep, sp)]
    variables, targets = joint['variables'], joint['targets']
    columns = [variables.index(t) for t in targets]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    stores = {g: SourceStore([s for s in shards if s['group'] == g and s['role'] == 'fit'], columns)
              for g in ('selected', 'broad')}
    cp = root / 'source_exposure_pair/expanded/real.pt'
    ck = torch.load(cp, weights_only=False)
    if ck['bindings'] != training['bindings']:
        raise ValueError('Checkpoint mismatch')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    params = list(model.trajectory.layer.body_update_gate.parameters())
    rng = np.random.default_rng(20260919)
    gradients, support = {}, {}
    levels = (8, 16, 32, 64)
    for group, store in stores.items():
        for c in columns:
            task = f'{group}/{variables[c]}'
            keys = store.groups[c]
            support[task] = len(keys)
            gradients[task] = {}
            for batch, order in enumerate(split_prefixes(len(keys), rng)):
                cumulative = None
                for i, pos in enumerate(order, 1):
                    e = store[keys[pos]]
                    pred = model(input_tokens(e, 'real', len(variables)), e['event_hours'], e['anchor_hour'], variables[c])
                    loss = torch.nn.functional.smooth_l1_loss(pred, pred.new_tensor([e['truth_normalized']]))
                    grad = torch.autograd.grad(loss, params)
                    vector = torch.cat([g.detach().flatten() for g in grad]).double().numpy()
                    if not np.isfinite(vector).all():
                        raise ValueError('Nonfinite gradient')
                    cumulative = vector if cumulative is None else cumulative + vector
                    if i in levels:
                        gradients[task].setdefault(i, {})[batch] = cumulative / i
            print('Probed', task, 'support', len(keys), flush=True)
    results = {}
    common = [t for t, g in gradients.items() if 64 in g]
    for n in levels:
        reliable = {t: cosine(g[n][0], g[n][1]) for t, g in gradients.items() if n in g}
        cross = {t: [cosine(gradients[f'selected/{t}'][n][b], gradients[f'broad/{t}'][n][b]) for b in (0, 1)]
                 for t in targets if all(n in gradients[f'{g}/{t}'] for g in stores)}
        values = [v for v in reliable.values() if v is not None]
        common_values = [reliable[t] for t in common if reliable[t] is not None]
        results[n] = dict(supported_tasks=len(reliable), within_task_cosine=reliable,
            median_cosine=float(np.median(values)), negative_tasks=sum(v < 0 for v in values),
            common64_tasks=len(common_values), common64_median=float(np.median(common_values)),
            common64_negative=sum(v < 0 for v in common_values), cross_group_cosine=cross,
            repeated_cross_conflicts=[t for t, v in cross.items() if all(x is not None and x < 0 for x in v)])
    out.write_text(json.dumps(dict(levels=results, support=support,
        bindings={str(p): sha256(p) for p in (Path(__file__), tp, ip, ep, sp, cp)},
        runtime_modified=False, limitation='One nested split, two disjoint source-fit pools per task. Local body gate only, not Adam updates or inference performance; common64 subset excludes sparse tasks.'), indent=2) + '\n')
    for n, r in results.items():
        print('SIZE', n, 'tasks', r['supported_tasks'], 'median', r['median_cosine'],
              'common64median', r['common64_median'], 'negative', r['negative_tasks'],
              'cross', r['repeated_cross_conflicts'])


if __name__ == '__main__':
    main()

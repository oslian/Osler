"""Two disjoint source probes of trained body-update gradients, no optimization."""
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


def geometry(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(matrix @ matrix.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return cosine, matrix @ matrix.mean(axis=0), norms


def main():
    out = Path('artifacts/native_reconstruction_trained_state_gradients_20260919.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    root = Path('data/research_recovery/broad_source_20260917')
    tp = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    train = json.loads(tp.read_text())
    for p, h in train['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Training changed')
    ip, ep = root / 'joint_source_manifest.pt', root / 'full_samples/index.pt'
    sp = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint, full, split = [torch.load(p, weights_only=False) for p in (ip, ep, sp)]
    variables, targets = joint['variables'], joint['targets']
    cols = [variables.index(t) for t in targets]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    stores = {g: SourceStore([s for s in shards if s['group'] == g and s['role'] == 'fit'], cols)
              for g in ('selected', 'broad')}
    cp = root / 'source_exposure_pair/expanded/real.pt'
    ck = torch.load(cp, weights_only=False)
    if ck['bindings'] != train['bindings']:
        raise ValueError('Checkpoint mismatch')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    named = [(n, p) for n, p in model.named_parameters() if n.startswith('trajectory.layer.body_update_gate.')]
    if not named:
        raise ValueError('Missing shared update parameters')
    params = [p for _, p in named]
    rng = np.random.default_rng(20260919)
    tasks, probes = [], [[], []]
    for group, store in stores.items():
        for col in cols:
            keys = store.groups[col]
            if len(keys) < 16:
                raise ValueError('Need two disjoint eight-patient probes')
            order = rng.permutation(len(keys))[:16]
            tasks.append(f'{group}/{variables[col]}')
            for batch in range(2):
                mean = None
                for pos in order[batch * 8:(batch + 1) * 8]:
                    e = store[keys[pos]]
                    prediction = model(input_tokens(e, 'real', len(variables)), e['event_hours'], e['anchor_hour'], variables[col])
                    loss = torch.nn.functional.smooth_l1_loss(prediction, prediction.new_tensor([e['truth_normalized']]))
                    grad = torch.autograd.grad(loss, params, allow_unused=False)
                    vector = torch.cat([g.detach().flatten() for g in grad]).double().numpy()
                    if not np.isfinite(vector).all():
                        raise ValueError('Nonfinite gradient')
                    mean = vector / 8 if mean is None else mean + vector / 8
                probes[batch].append(mean)
        print('Gradient group complete', group, flush=True)
    arrays = [np.stack(p) for p in probes]
    results = [geometry(a) for a in arrays]
    cross = {t: [float(r[0][i, len(targets) + i]) for r in results] for i, t in enumerate(targets)}
    shared_harm = [tasks[i] for i in range(len(tasks)) if all(r[1][i] < 0 for r in results)]
    reliability = np.sum(arrays[0] * arrays[1], axis=1) / np.maximum(results[0][2] * results[1][2], 1e-30)
    report = dict(tasks=tasks, parameters=[n for n, _ in named],
        parameter_count=sum(p.numel() for p in params), patients_per_task_per_probe=8,
        same_target_cross_group_cosine=cross, average_gradient_first_order_harm_both=shared_harm,
        within_task_probe_cosine={t: float(v) for t, v in zip(tasks, reliability)},
        probes=[dict(cosine=r[0].tolist(), dot_with_average=r[1].tolist(), norm=r[2].tolist()) for r in results],
        bindings={str(p): sha256(p) for p in (Path(__file__), tp, ip, ep, sp, cp)},
        runtime_modified=False, limitation='Local body-update gradients, two small disjoint source-fit probes; SGD first-order geometry is not Adam intervention, full encoder conflict or generalization proof.')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('Cross-group negative in both:', [t for t, v in cross.items() if max(v) < 0])
    print('Average gradient harms in both:', shared_harm)
    print('Median within-task probe cosine:', float(np.median(reliability)))


if __name__ == '__main__':
    main()

"""Decay-only controlled intervention on the fixed accumulated-source protocol."""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_source_exposure_pair import epoch_plan
from scripts.audit_optimizer_exposure_clock import matched_decay
from scripts.train_joint_source_reconstruction import equal_group_median
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def accumulated_step(model, optimizer, examples, loss_fn):
    if not examples:
        raise ValueError('Nonempty accumulated batch required')
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for example in examples:
        loss = loss_fn(model, example)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        (loss / len(examples)).backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return total


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_decay_matched_source_20260919.json')
    if out.exists():
        raise ValueError('Completed experiment exists')
    tp = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    baseline = json.loads(tp.read_text())
    for p, h in baseline['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Baseline changed')
    ip, ep = root / 'joint_source_manifest.pt', root / 'full_samples/index.pt'
    sp = Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint, full, split = [torch.load(p, weights_only=False) for p in (ip, ep, sp)]
    variables, targets = joint['variables'], joint['targets']
    cols = [variables.index(t) for t in targets]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    data = {g: {r: SourceStore([s for s in shards if s['group'] == g and s['role'] == r], cols)
                for r in ('fit', 'encoder_tune')} for g in ('selected', 'broad')}
    groups = {g: d['fit'].groups for g, d in data.items()}
    medians = {c: equal_group_median([np.array([data[g]['fit'].labels[k] for k in groups[g][c]]) for g in data]) for c in cols}
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(19)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    initial = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_(); initial.readout.bias.zero_()
        for c, value in medians.items():
            initial.readout.bias[c] = value
    batch_size = 8 * len(cols) * len(data)
    paths = [Path(__file__), tp, ip, ep, sp, Path('RECONSTRUCTION_DECAY_PROTOCOL_20260919.md'),
             Path('scripts/train_source_exposure_pair.py'), Path('scripts/train_accumulated_source.py'),
             Path('scripts/audit_optimizer_exposure_clock.py')]
    bindings = {str(p): sha256(p) for p in paths}

    def loss_fn(model, item):
        group, key = item
        e = data[group]['fit'][key]
        p = model(input_tokens(e, 'real', len(variables)), e['event_hours'], e['anchor_hour'], variables[key[1]])
        return torch.nn.functional.smooth_l1_loss(p, p.new_tensor([e['truth_normalized']]))

    def score(model):
        result = {}
        with torch.inference_mode():
            for g, d in data.items():
                errors = {c: [] for c in cols}
                for key, truth in d['encoder_tune'].labels.items():
                    e = d['encoder_tune'][key]
                    p = float(model(input_tokens(e, 'real', len(variables)), e['event_hours'], e['anchor_hour'], variables[key[1]])[0])
                    errors[key[1]].append(abs(p - truth))
                result[g] = {variables[c]: float(np.mean(v)) for c, v in errors.items()}
        value = float(np.mean([v for g in result.values() for v in g.values()]))
        if not np.isfinite(value):
            raise ValueError('Invalid selection score')
        return value, result

    summary = {}
    private = root / 'decay_matched_training'
    private.mkdir(exist_ok=True)
    for arm, lr in (('decay_matched', 1e-4 * math.sqrt(batch_size)),):
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=matched_decay(1e-4, .01, batch_size, lr))
        rng = np.random.default_rng(19)
        queues = {g: {c: {} for c in cols} for g in data}
        path = private / f'{arm}.pt'
        if path.exists():
            ck = torch.load(path, weights_only=False)
            if ck['bindings'] != bindings:
                raise ValueError('Resume mismatch')
            model.load_state_dict(ck['state_dict']); optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state = ck['rng']; torch.set_rng_state(ck['torch_rng']); queues = ck['queues']
        else:
            value, per = score(model)
            if per != baseline['modes']['expanded']['real']['initial_scores']:
                raise ValueError('Baseline initialization mismatch')
            ck = dict(epoch=0, selected_epoch=0, best_tune=value, best_state_dict=deepcopy(model.state_dict()),
                      initial_scores=per, log=[], bindings=bindings)
        for epoch in range(ck['epoch'] + 1, 9):
            plan = epoch_plan(groups, queues, rng)
            if len(plan) != baseline['modes']['expanded']['real']['log'][epoch-1]['updates']:
                raise ValueError('Unequal exposure budget')
            total = 0.; steps = 0
            for start in range(0, len(plan), batch_size):
                block = plan[start:start+batch_size]
                if len(block) != batch_size:
                    raise ValueError('Unbalanced partial batch')
                total += accumulated_step(model, optimizer, block, loss_fn); steps += 1
            value, per = score(model)
            ck['log'].append(dict(epoch=epoch, tune_mae=value, group_target_scores=per,
                                 examples=len(plan), optimizer_steps=steps, draw_loss=total/len(plan)))
            if value < ck['best_tune']:
                ck.update(best_tune=value, selected_epoch=epoch, best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                      torch_rng=torch.get_rng_state(), queues=queues, variables=variables, targets=targets, medians=medians)
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, bindings=bindings), private/f'{arm}_epoch{epoch}.pt')
            temp = path.with_suffix('.tmp'); torch.save(ck, temp); temp.replace(path)
            print(arm, epoch, value, 'selected', ck['selected_epoch'], flush=True)
        summary[arm] = {k: ck[k] for k in ('epoch', 'selected_epoch', 'best_tune', 'initial_scores', 'log')}
        summary[arm]['lr'] = lr
        summary[arm]['weight_decay'] = optimizer.param_groups[0]['weight_decay']
    choice = min(summary, key=lambda k: summary[k]['best_tune'])
    out.write_text(json.dumps(dict(training_complete=True, arms=summary, source_selected_arm=choice,
        batch_size=batch_size, source_targets=targets, bindings=bindings, runtime_modified=False, promotion_allowed=False), indent=2)+'\n')


if __name__ == '__main__':
    main()


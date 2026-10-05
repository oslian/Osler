"""E39: replay E38 exactly and account for local optimizer-stage directions."""
import json
import importlib
import inspect
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts import train_pair_gradient as parent
from whole_body_jepa.multisource_intervals import sha256

STAGES = ('proposal', 'momentum', 'preconditioned', 'with_decay', 'actual')
DEST = parent.ROOT / 'adam_stages_20261005'
OUTPUT = Path('artifacts/native_adam_stages_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_ADAM_STAGES_PROTOCOL_20261005.md')


def adam_stages(before, after, proposal, state, group):
    """Read actual moments; this function never updates parameters or moments."""
    arrays = [x.detach().double().clone() for x in
              (before, after, proposal, state['exp_avg'], state['exp_avg_sq'])]
    b, a, q, m, v = arrays
    if (not b.numel() or any(x.shape != b.shape or not torch.isfinite(x).all() for x in arrays)
            or (v < 0).any()):
        raise ValueError('Invalid parameters or moments')
    if (any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable'))
            or not group.get('decoupled_weight_decay', True)):
        raise ValueError('Unsupported Adam variant')
    t = float(state['step'])
    lr, wd, eps = float(group['lr']), float(group['weight_decay']), float(group['eps'])
    b1, b2 = (float(x) for x in group['betas'])
    if (not np.isfinite([t, lr, wd, eps, b1, b2]).all() or t < 1 or t != int(t)
            or lr <= 0 or wd < 0 or eps <= 0 or not 0 <= b1 < 1 or not 0 <= b2 < 1):
        raise ValueError('Invalid optimizer settings')
    moment = -lr * m / (1 - b1 ** int(t))
    scaled = moment / (torch.sqrt(v / (1 - b2 ** int(t))) + eps)
    result = dict(proposal=-lr*q, momentum=moment, preconditioned=scaled,
                  with_decay=scaled-lr*wd*b, actual=a-b)
    if any(not torch.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite stage')
    return result


def gradient_metrics(gradient, stages):
    if tuple(stages) != STAGES:
        raise ValueError('Changed stages')
    g = gradient.detach().double()
    if not torch.isfinite(g).all() or not g.numel():
        raise ValueError('Invalid gradient')
    result = {}
    for name, delta in stages.items():
        if delta.shape != g.shape or not torch.isfinite(delta).all():
            raise ValueError('Invalid step')
        result[name+'_dot'] = float((g*delta).sum())
        result[name+'_tolerance'] = 1e-12*(1+float(g.norm()*delta.norm()))
    return result


def trace_fit(original, examples, bound, arm):
    """Mirror the immutable E38 loop; exact final tensors/logs are mandatory."""
    if arm not in parent.ARMS or not examples or len(examples) != len(bound) or len(examples) % 2:
        raise ValueError('Aligned even nonempty fit required')
    torch.manual_seed(parent.SEED)
    model = parent.AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
    weight = model.bridge.projection.weight
    weight.requires_grad_(True)
    opt = torch.optim.AdamW([weight], lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(parent.SEED)
    log, rows, max_rounding = [], [], 0.
    for epoch in range(2):
        order = rng.permutation(len(examples)).reshape(-1, 2)
        losses = []
        stats = dict(steps=0, conflicting_pairs=0, projected_pairs=0,
                     proposed_harmful_dots=0, actual_harmful_dots=0,
                     maximum_actual_harm=0., maximum_proposed_harm=0.)
        for step, pair in enumerate(order):
            grads, targets, inputs, prior = [], [], [], []
            for i in pair:
                e = parent.select_arm(examples[i], bound[i], 'binding')
                pred = model(e)
                truth = pred.new_tensor(e['truth'])
                loss = F.smooth_l1_loss(pred, truth)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                g, = torch.autograd.grad(loss, weight)
                grads.append(g.detach()); targets.append(e['target']); inputs.append(e)
                losses.append(float(loss.detach()))
                prior.append((float(pred.detach()), float(truth), float(loss.detach()),
                              float((pred-truth).abs().detach())))
            conflict = float((grads[0].double()*grads[1].double()).sum()) < 0
            surgery = arm == 'pcgrad' and targets[0] != targets[1]
            combined = parent.combine(*grads, surgery)
            opt.zero_grad(set_to_none=True); weight.grad = combined
            torch.nn.utils.clip_grad_norm_([weight], 1., error_if_nonfinite=True)
            proposal = weight.grad.detach().double().clone()
            before = weight.detach().double().clone()
            opt.step()
            delta = weight.detach().double()-before
            if not torch.isfinite(weight).all():
                raise ValueError('Nonfinite updated weights')
            stages = adam_stages(before, weight, proposal, opt.state[weight], opt.param_groups[0])
            rounding = float((stages['actual']-stages['with_decay']).abs().max())
            limit = 64*torch.finfo(weight.dtype).eps*(1+float(before.abs().max())+
                                                    float(stages['with_decay'].abs().max()))
            if rounding > limit:
                raise ValueError('Optimizer accounting exceeds roundoff bound')
            max_rounding = max(max_rounding, rounding)
            for slot, (g, e, old) in enumerate(zip(grads, inputs, prior)):
                g = g.double()
                dp = -float((g*proposal).sum()); da = float((g*delta).sum())
                stats['proposed_harmful_dots'] += int(dp > 1e-12*(1+float(g.norm()*proposal.norm())))
                stats['actual_harmful_dots'] += int(da > 1e-12*(1+float(g.norm()*delta.norm())))
                stats['maximum_proposed_harm'] = max(stats['maximum_proposed_harm'], dp)
                stats['maximum_actual_harm'] = max(stats['maximum_actual_harm'], da)
                with torch.no_grad():
                    pred = model(e); truth = pred.new_tensor(e['truth'])
                    new_loss = float(F.smooth_l1_loss(pred, truth))
                    new_mae = float((pred-truth).abs())
                rows.append(dict(epoch=epoch+1, step=step, slot=slot, target=e['target'],
                    same_target=targets[0] == targets[1], projected=bool(conflict and surgery),
                    prediction_before=old[0], prediction_after=float(pred), truth=old[1],
                    loss_before=old[2], loss_after=new_loss, mae_before=old[3], mae_after=new_mae,
                    unscaled_proposal_dot=dp,
                    unscaled_proposal_tolerance=1e-12*(1+float(g.norm()*proposal.norm())),
                    maximum_coordinate_roundoff=rounding, **gradient_metrics(g, stages)))
            stats['steps'] += 1; stats['conflicting_pairs'] += int(conflict)
            stats['projected_pairs'] += int(conflict and surgery)
            if (step+1) % 512 == 0:
                print('E39 exact replay', arm, epoch+1, step+1, len(order), flush=True)
        parent.assert_frozen(model, original)
        log.append(dict(epoch=epoch+1, examples=len(losses), loss=float(np.mean(losses)), **stats))
    return model, log, pd.DataFrame(rows), max_rounding


def summarize(frame):
    if frame.empty:
        return dict(status='unsupported', examples=0)
    keys = [s+x for s in STAGES for x in ('_dot', '_tolerance')]
    keys += ['prediction_before', 'prediction_after', 'truth', 'loss_before', 'loss_after', 'mae_before', 'mae_after',
             'unscaled_proposal_dot', 'unscaled_proposal_tolerance', 'maximum_coordinate_roundoff']
    if not np.isfinite(frame[keys].to_numpy()).all() or (frame[[s+'_tolerance' for s in STAGES]] <= 0).any().any():
        raise ValueError('Nonfinite or invalid diagnostic rows')
    harmful = np.column_stack([(frame[s+'_dot'] > frame[s+'_tolerance']).to_numpy() for s in STAGES])
    first = np.where(harmful.any(axis=1), harmful.argmax(axis=1), len(STAGES))
    names = (*STAGES, 'none')
    actual = harmful[:, -1]
    loss_delta = (frame.loss_after-frame.loss_before).to_numpy()
    loss_tol = 1e-12*(1+frame.loss_before.abs()+frame.loss_after.abs()).to_numpy()
    finite_harm = loss_delta > loss_tol
    finite_help = loss_delta < -loss_tol
    result = dict(status='scored', examples=len(frame),
        harmful={s:int(harmful[:, i].sum()) for i, s in enumerate(STAGES)},
        first_harm={s:int((first == i).sum()) for i, s in enumerate(names)},
        first_harm_among_actual_harm={s:int(((first == i)&actual).sum()) for i, s in enumerate(names)},
        stage_transitions={b:dict(safe_to_harm=int((~harmful[:, i]&harmful[:, i+1]).sum()),
            harm_to_safe=int((harmful[:, i]&~harmful[:, i+1]).sum()))
            for i, (a, b) in enumerate(zip(STAGES[:-1], STAGES[1:]))},
        finite_loss=dict(harmful=int(finite_harm.sum()), helpful=int(finite_help.sum()),
            ambiguous=int((~finite_harm&~finite_help).sum()),
            actual_dot_harm_and_finite_harm=int((actual&finite_harm).sum()),
            actual_dot_safe_but_finite_harm=int((~actual&finite_harm).sum())),
        mean_loss_change=float(loss_delta.mean()),
        mean_mae_change=float((frame.mae_after-frame.mae_before).mean()),
        unscaled_proposal_harm=int((frame.unscaled_proposal_dot > frame.unscaled_proposal_tolerance).sum()),
        maximum_coordinate_roundoff=float(frame.maximum_coordinate_roundoff.max()))
    # Independent scalar counts, including transitions and finite changes.
    flags = [[float(getattr(r, s+'_dot')) > float(getattr(r, s+'_tolerance')) for s in STAGES]
             for r in frame.itertuples(index=False)]
    for i, s in enumerate(STAGES):
        if result['harmful'][s] != sum(x[i] for x in flags):
            raise ValueError('Scalar stage-count mismatch')
        first_count = sum(next((j for j, v in enumerate(x) if v), len(STAGES)) == i for x in flags)
        if result['first_harm'][s] != first_count:
            raise ValueError('Scalar first-harm mismatch')
        actual_first = sum(x[-1] and next((j for j, v in enumerate(x) if v), len(STAGES)) == i for x in flags)
        if result['first_harm_among_actual_harm'][s] != actual_first:
            raise ValueError('Scalar actual-harm origin mismatch')
    for i, s in enumerate(STAGES[1:]):
        expected = dict(safe_to_harm=sum(not x[i] and x[i+1] for x in flags),
                        harm_to_safe=sum(x[i] and not x[i+1] for x in flags))
        if result['stage_transitions'][s] != expected:
            raise ValueError('Scalar transition mismatch')
    for field, before, after in (('mean_loss_change','loss_before','loss_after'),
                                 ('mean_mae_change','mae_before','mae_after')):
        check = fsum(float(getattr(r, after))-float(getattr(r, before))
                     for r in frame.itertuples(index=False))/len(frame)
        if abs(check-result[field]) > 1e-12*(1+abs(check)):
            raise ValueError('Scalar mean mismatch')
    scalar_harm, scalar_help = [], []
    for r in frame.itertuples(index=False):
        change = float(r.loss_after)-float(r.loss_before)
        tol = 1e-12*(1+abs(float(r.loss_after))+abs(float(r.loss_before)))
        scalar_harm.append(change > tol); scalar_help.append(change < -tol)
    finite_check = dict(harmful=sum(scalar_harm), helpful=sum(scalar_help),
        ambiguous=sum(not h and not b for h,b in zip(scalar_harm,scalar_help)),
        actual_dot_harm_and_finite_harm=sum(x[-1] and h for x,h in zip(flags,scalar_harm)),
        actual_dot_safe_but_finite_harm=sum(not x[-1] and h for x,h in zip(flags,scalar_harm)))
    if result['finite_loss'] != finite_check:
        raise ValueError('Scalar finite-loss mismatch')
    if sum(result['first_harm'].values()) != len(frame) or sum(result['first_harm_among_actual_harm'].values()) != int(actual.sum()):
        raise ValueError('Incomplete classification')
    return result


def main():
    if OUTPUT.exists():
        raise ValueError('Completed diagnostic exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    verification = Path('artifacts/native_pair_gradient_verification_20261005.json')
    report = json.loads(parent.OUTPUT.read_text())
    verified = json.loads(verification.read_text())
    if not report['complete'] or not verified['complete']:
        raise ValueError('Incomplete E38')
    parent.check_bindings(report['bindings']); parent.check_bindings(verified['bindings'])
    paths = [PROTOCOL, Path(__file__), Path('tests/test_adam_stages.py'), parent.OUTPUT, verification]
    paths += [Path(inspect.getsourcefile(importlib.import_module('torch.optim.'+name)))
              for name in ('adam','adamw')]
    bindings = {str(p):sha256(p) for p in paths}
    all_frames, inventory, exact = [], None, []
    for fold in range(5):
        for directory, suffix in (('content_training_20261003',''), ('organ_binding_20261004','_binding')):
            for role in ('fit', 'selection'):
                for ext in ('joblib', 'json'):
                    if not (parent.ROOT/directory/f'fold{fold}'/f'{role}{suffix}.{ext}').is_file():
                        raise ValueError('Missing immutable source cache')
        variables, examples, bound, original, contract, extraction, _ = parent.prepare(fold)
        if len(examples['fit']) != 2048 or len(variables) != 53 or (inventory is not None and variables != inventory):
            raise ValueError('Changed inventory or fitting support')
        inventory = variables
        source = parent.DEST/f'fold{fold}'
        cp = source/'contract.json'; old = json.loads(cp.read_text())
        parent.check_bindings(old['bindings'])
        if old['parent'] != contract or old['extraction'] != extraction:
            raise ValueError('Changed fitting contract')
        for arm in parent.ARMS:
            checkpoint, metadata = source/f'{arm}.pt', source/f'{arm}.json'
            info = json.loads(metadata.read_text())
            for p in (cp, checkpoint, metadata):
                if report['bindings'][str(p)] != sha256(p):
                    raise ValueError('Changed E38 dependency')
            if info['sha256'] != sha256(checkpoint) or info['contract'] != sha256(cp) or info['arm'] != arm:
                raise ValueError('Invalid E38 model contract')
            dependencies = dict(bindings, **{str(p):sha256(p) for p in (cp, checkpoint, metadata)})
            dest = DEST/f'fold{fold}'; dest.mkdir(parents=True, exist_ok=True)
            p, mp = dest/f'{arm}.parquet', dest/f'{arm}.json'
            if mp.exists():
                done = json.loads(mp.read_text())
                if done['bindings'] != dependencies or done['sha256'] != sha256(p) or not done['tensors_exact']:
                    raise ValueError('Changed completed trace')
                frame = pd.read_parquet(p)
            else:
                if p.exists():
                    raise ValueError('Unverified trace')
                parent.space_required()
                model, log, frame, rounding = trace_fit(original, examples['fit'], bound['fit'], arm)
                saved = torch.load(checkpoint, weights_only=False)
                if set(saved) != set(model.state_dict()) or log != info['training']:
                    raise ValueError('Original E38 training not reproduced')
                for k, value in model.state_dict().items():
                    torch.testing.assert_close(value, saved[k], atol=0, rtol=0)
                if len(frame) != 4096 or frame.duplicated(['epoch','step','slot']).any():
                    raise ValueError('Incomplete trace')
                frame['fold'] = fold; frame['arm'] = arm
                frame['variable'] = [variables[int(x)] for x in frame.target]
                summary = summarize(frame)
                if (summary['unscaled_proposal_harm'] != sum(x['proposed_harmful_dots'] for x in log)
                        or summary['harmful']['actual'] != sum(x['actual_harmful_dots'] for x in log)):
                    raise ValueError('Original E38 counts not reproduced')
                temp = p.with_suffix('.tmp'); frame.to_parquet(temp, index=False); temp.replace(p)
                parent.write_json(mp, dict(bindings=dependencies, sha256=sha256(p), training=log,
                    tensors_exact=True, summary=summary, maximum_coordinate_roundoff=rounding))
            if len(frame) != 4096 or set(frame.fold) != {fold} or set(frame.arm) != {arm}:
                raise ValueError('Changed trace role')
            all_frames.append(frame); exact.append(dict(fold=fold, arm=arm, tensors_and_logs_exact=True))
            print('E39 verified', fold, arm, summarize(frame)['harmful'], flush=True)
    frame = pd.concat(all_frames, ignore_index=True)
    if len(frame) != 40960 or frame.duplicated(['fold','arm','epoch','step','slot']).any():
        raise ValueError('Incomplete experiment')
    result = {arm:dict(all=summarize(f),
        same_target=summarize(f[f.same_target]), cross_target=summarize(f[~f.same_target]),
        targets={v:summarize(f[f.variable == v]) for v in inventory})
        for arm, f in frame.groupby('arm')}
    for fold in range(5):
        for arm in parent.ARMS:
            for ext in ('json','parquet'):
                p = DEST/f'fold{fold}'/f'{arm}.{ext}'; bindings[str(p)] = sha256(p)
    parent.check_bindings(report['bindings']); parent.check_bindings(verified['bindings']); parent.check_bindings(bindings)
    parent.write_json(OUTPUT, dict(complete=True, bindings=bindings, exact_replay=exact,
        examples=len(frame), inventory=inventory, results=result, torch_version=torch.__version__,
        runtime_modified=False, promotion_allowed=False,
        limitation='Local conditional training-step accounting; not optimizer removal causality or unseen-patient repair.'))
    print(json.dumps({a:r['all'] for a, r in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()

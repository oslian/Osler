"""E40: project the actual Adam step, retaining the E38 training contract."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts import train_pair_gradient as base

DEST = base.ROOT/'actual_step_20261005'
OUTPUT = Path('artifacts/native_actual_step_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_ACTUAL_STEP_PROTOCOL_20261005.md')
CONTROLS = ('pcgrad', *base.CONTROLS)
PRIMARY = 'actual_step'


def harmful(g, step):
    g, step = g.detach().double(), step.detach().double()
    return float((g*step).sum()) > 1e-12*(1+float(g.norm()*step.norm()))


def project_step(step, gradients):
    """Nearest point in two homogeneous half-spaces; all algebra is float64."""
    if (len(gradients) != 2 or not step.numel() or not torch.isfinite(step).all()
            or any(g.shape != step.shape or not torch.isfinite(g).all() for g in gradients)):
        raise ValueError('Invalid step or paired gradients')
    d = step.detach().double().flatten()
    normals = [g.detach().double().flatten() for g in gradients]
    normals = [g/g.norm() for g in normals if float(g.norm()) > 0]
    if not normals:
        return step.detach().double().clone()
    a = torch.stack(normals)
    tolerance = 64*torch.finfo(torch.float64).eps*(1+float(d.norm()))
    if bool((a@d <= 0).all()):
        return step.detach().double().clone()
    candidates = [torch.zeros_like(d)]
    for n in a:
        v = d-n*torch.dot(n, d)
        if bool((a@v <= tolerance).all()):
            candidates.append(v)
    if len(a) == 2:
        # Handles parallel/opposing constraints without inverting a Gram matrix.
        coefficients = torch.linalg.lstsq(a.T, d, driver='gelsd').solution
        v = d-a.T@coefficients
        if bool((a@v <= tolerance).all()):
            candidates.append(v)
    answer = min(candidates, key=lambda v: float((v-d).square().sum()))
    if not torch.isfinite(answer).all() or not bool((a@answer <= tolerance).all()):
        raise ValueError('Infeasible projected step')
    return answer.reshape(step.shape)


def corrected_weight(before, proposed, gradients):
    """Correct only when needed; enforce the constraint after dtype rounding."""
    delta = proposed.double()-before.double()
    if not any(harmful(g, delta) for g in gradients):
        return proposed.clone(), False, False
    projected = project_step(delta, gradients)
    candidate = (before.double()+projected).to(before.dtype)
    veto = any(harmful(g, candidate.double()-before.double()) for g in gradients)
    return (before.clone() if veto else candidate), True, veto


def fit(original, examples, bound, protect=True):
    if not examples or len(examples) != len(bound) or len(examples) % 2:
        raise ValueError('Aligned even nonempty fit required')
    torch.manual_seed(base.SEED)
    model = base.AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
    weight = model.bridge.projection.weight
    weight.requires_grad_(True)
    opt = torch.optim.AdamW([weight], lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(base.SEED)
    logs, trace = [], []
    for epoch in range(2):
        order = rng.permutation(len(examples)).reshape(-1, 2)
        losses = []
        for step, pair in enumerate(order):
            es = [base.select_arm(examples[i], bound[i], 'binding') for i in pair]
            grads, predictions, previous = [], [], []
            for e in es:
                pred = model(e)
                loss = F.smooth_l1_loss(pred, pred.new_tensor(e['truth']))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite fit loss')
                g, = torch.autograd.grad(loss, weight)
                grads.append(g.detach()); predictions.append(float(pred.detach()))
                previous.append(float(loss.detach())); losses.append(float(loss.detach()))
            cross = es[0]['target'] != es[1]['target']
            opt.zero_grad(set_to_none=True)
            weight.grad = base.combine(*grads, cross)
            torch.nn.utils.clip_grad_norm_([weight], 1., error_if_nonfinite=True)
            before = weight.detach().clone()
            opt.step()
            proposed = weight.detach().clone()
            projected, veto = False, False
            if protect and cross:
                value, projected, veto = corrected_weight(before, proposed, grads)
                with torch.no_grad():
                    weight.copy_(value)
            raw, actual = proposed.double()-before.double(), weight.detach().double()-before.double()
            if not torch.isfinite(weight).all():
                raise ValueError('Nonfinite weights')
            with torch.no_grad():
                for slot, (e, g) in enumerate(zip(es, grads)):
                    pred = model(e)
                    loss = F.smooth_l1_loss(pred, pred.new_tensor(e['truth']))
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite updated fit loss')
                    tol = 1e-12*(1+float(g.double().norm()*actual.norm()))
                    dot = float((g.double()*actual).sum())
                    if protect and cross and dot > tol:
                        raise ValueError('Stored step violated local constraint')
                    trace.append(dict(epoch=epoch+1, step=step, slot=slot,
                        target=e['target'], cross_target=cross, projected=projected, rounding_veto=veto,
                        raw_dot=float((g.double()*raw).sum()),
                        raw_tolerance=1e-12*(1+float(g.double().norm()*raw.norm())),
                        actual_dot=dot, actual_tolerance=tol,
                        raw_norm=float(raw.norm()), actual_norm=float(actual.norm()),
                        correction_norm=float((actual-raw).norm()), truth=float(e['truth']),
                        prediction_before=predictions[slot], prediction_after=float(pred),
                        loss_before=previous[slot], loss_after=float(loss)))
            if (step+1) % 512 == 0:
                print('E40 fit', epoch+1, step+1, len(order), flush=True)
        base.assert_frozen(model, original)
        logs.append(dict(epoch=epoch+1, examples=len(losses), steps=len(order), loss=float(np.mean(losses))))
    return model, logs, pd.DataFrame(trace)


def describe(f, inventory):
    base.validate_identities(f)
    names = [n for n in (PRIMARY, *CONTROLS) if n in f]
    if not {PRIMARY, 'pcgrad', 'mean'} <= set(names) or not set(f.variable) <= set(inventory):
        raise ValueError('Missing candidate/control or invalid inventory')
    if not np.isfinite(f[['truth', *names]].to_numpy()).all():
        raise ValueError('Nonfinite values')
    results = {}
    for variable in inventory:
        part = f[f.variable == variable]
        if part.empty:
            results[variable] = dict(status='unsupported'); continue
        people = part[names].subtract(part.truth, axis=0).abs().groupby(part.patient).mean()
        links = part.drop_duplicates('patient').set_index('patient').component
        groups = people.groupby(links.reindex(people.index)).mean()
        results[variable] = dict(status='scored', patients=len(people), components=len(groups), occurrences=len(part),
            normalized_mae=dict(patient=people.mean().to_dict(), component=groups.mean().to_dict()))
    return results


def assess(f, seed):
    if f.variable.nunique() != 1 or not set(CONTROLS) <= set(f):
        raise ValueError('Invalid assessment')
    entry = describe(f, [f.variable.iloc[0]])[f.variable.iloc[0]]
    people = f[[PRIMARY, *CONTROLS]].subtract(f.truth, axis=0).abs().groupby(f.patient).mean()
    links = f.drop_duplicates('patient').set_index('patient').component
    groups = people.groupby(links.reindex(people.index)).mean()
    entry['comparisons'] = {}
    for i, control in enumerate(CONTROLS):
        stats = {level: base.paired_summary(table[PRIMARY]-table[control], seed+i+offset, 5000)
                 for level, table, offset in (('patient', people, 0), ('component', groups, 100))}
        passed = len(people) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*len(CONTROLS)) for s in stats.values())
        entry['comparisons'][control] = dict(paired=stats, screen_pass=bool(passed))
    return entry


def trace_summary(f):
    if f.empty or f.duplicated(['fold', 'epoch', 'step', 'slot']).any():
        raise ValueError('Invalid training trace')
    result = {}
    for name, p in (('all', f), ('cross_target', f[f.cross_target]), ('same_target', f[~f.cross_target])):
        change = p.loss_after-p.loss_before
        tol = 1e-12*(1+p.loss_before.abs()+p.loss_after.abs())
        steps = p.drop_duplicates(['fold', 'epoch', 'step'])
        result[name] = dict(examples=len(p), steps=len(steps), projected_steps=int(steps.projected.sum()),
            rounding_veto_steps=int(steps.rounding_veto.sum()), raw_harm=int((p.raw_dot > p.raw_tolerance).sum()),
            actual_harm=int((p.actual_dot > p.actual_tolerance).sum()), finite_harm=int((change > tol).sum()),
            finite_help=int((change < -tol).sum()), finite_tie=int((change.abs() <= tol).sum()),
            mean_loss_change=float(change.mean()) if len(p) else None,
            mean_mae_change=float(((p.prediction_after-p.truth).abs()-(p.prediction_before-p.truth).abs()).mean()) if len(p) else None,
            mean_raw_norm=float(steps.raw_norm.mean()) if len(steps) else None,
            mean_actual_norm=float(steps.actual_norm.mean()) if len(steps) else None)
    return result


def prepare_existing(fold):
    for stage, suffix in (('content_training_20261003', ''), ('organ_binding_20261004', '_binding')):
        for role in ('fit', 'selection'):
            for ext in ('joblib', 'json'):
                if not (base.ROOT/stage/f'fold{fold}'/f'{role}{suffix}.{ext}').is_file():
                    raise ValueError('Missing original cache')
    return base.prepare(fold)


def run_fold(fold, parent, common):
    variables, examples, bound, original, contract, extraction, _ = prepare_existing(fold)
    olddir = base.DEST/f'fold{fold}'
    oldcp = olddir/'contract.json'
    if base.sha256(oldcp) != parent['bindings'][str(oldcp)]:
        raise ValueError('Changed E38 contract')
    old = json.loads(oldcp.read_text())
    if old['parent'] != contract or old['extraction'] != extraction or len(variables) != 53 or len(examples['fit']) != 2048:
        raise ValueError('Changed data contract')
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True, exist_ok=True)
    cp, path, meta_path, trace_path = [dest/name for name in ('contract.json', 'actual_step.pt', 'actual_step.json', 'training.parquet')]
    current = dict(bindings=common, parent=contract, extraction=extraction, old_contract=base.sha256(oldcp))
    if cp.exists() and json.loads(cp.read_text()) != current:
        raise ValueError('Changed experiment')
    if not cp.exists(): base.write_json(cp, current)
    if not meta_path.exists():
        if path.exists() or trace_path.exists(): raise ValueError('Unverified partial fit')
        base.space_required()
        model, logs, trace = fit(original, examples['fit'], bound['fit'])
        trace['fold'] = fold
        temp = path.with_suffix('.tmp'); torch.save(model.state_dict(), temp); temp.replace(path)
        temp = trace_path.with_suffix('.tmp'); trace.to_parquet(temp, index=False); temp.replace(trace_path)
        base.write_json(meta_path, dict(contract=base.sha256(cp), sha256=base.sha256(path),
            trace_sha256=base.sha256(trace_path), training=logs, trace_summary=trace_summary(trace)))
    meta = json.loads(meta_path.read_text())
    if meta['contract'] != base.sha256(cp) or meta['sha256'] != base.sha256(path) or meta['trace_sha256'] != base.sha256(trace_path):
        raise ValueError('Changed saved model')
    model = base.AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
    model.load_state_dict(torch.load(path, weights_only=False), strict=True)
    base.assert_frozen(model, original)
    reports = [cp, path, meta_path, trace_path]
    frames, patients = {}, {}
    for role in base.ROLES:
        src = olddir/f'{role}.parquet'
        if base.sha256(src) != parent['bindings'][str(src)]: raise ValueError('Changed controls')
        prior = pd.read_parquet(src); base.validate_identities(prior)
        patients[role] = set(prior.patient)
        if len(prior) != (2048 if role == 'fit' else 1024): raise ValueError('Changed role size')
        p, mp = dest/f'{role}.parquet', dest/f'{role}.json'
        bindings = {str(x): base.sha256(x) for x in (src, cp, path, meta_path)}
        if mp.exists():
            m = json.loads(mp.read_text())
            if m['bindings'] != bindings or m['sha256'] != base.sha256(p): raise ValueError('Changed completed role')
            f = pd.read_parquet(p)
        else:
            if p.exists(): raise ValueError('Unverified role')
            if role == 'same_source': rows = base.same_source_examples(prior, original, variables, fold)
            else:
                if list(prior[['patient','component','context','truth']].itertuples(index=False,name=None)) != [
                        (e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]:
                    raise ValueError('Changed identity/truth')
                rows = (base.select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
            points = []; parity = 0.
            with torch.inference_mode():
                for i, e in enumerate(rows):
                    value = float(model(e)); check = base.direct_prediction(model,e,e['organs'])
                    np.testing.assert_allclose(value,check,atol=1e-7,rtol=0)
                    points.append(value); parity = max(parity,abs(value-check))
                    if (i+1)%512 == 0: print('E40 replay',fold,role,i+1,len(prior),flush=True)
            if len(points) != len(prior): raise ValueError('Incomplete query replay')
            f = prior.copy(); f[PRIMARY] = points
            describe(f,variables); base.space_required()
            temp = p.with_suffix('.tmp'); f.to_parquet(temp,index=False); temp.replace(p)
            base.write_json(mp,dict(bindings=bindings,sha256=base.sha256(p),rows=len(f),parity=parity))
        pd.testing.assert_frame_equal(f[list(prior)],prior,check_exact=True)
        frames[role] = f; reports.extend((p,mp))
    if any(patients[a]&patients[b] for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Role overlap')
    return frames, variables, reports, pd.read_parquet(trace_path)


def main():
    if OUTPUT.exists(): raise ValueError('Completed experiment exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    parent = json.loads(base.OUTPUT.read_text())
    verify = Path('artifacts/native_pair_gradient_verification_20261005.json')
    cert = json.loads(verify.read_text())
    if not parent['complete'] or not cert['complete'] or cert['bindings'][str(base.OUTPUT)] != base.sha256(base.OUTPUT):
        raise ValueError('Missing verified E38')
    base.check_bindings(parent['bindings']); base.check_bindings(cert['bindings'])
    e39 = Path('artifacts/native_adam_stages_verification_20261005.json')
    stage = json.loads(e39.read_text())
    if not stage['complete']: raise ValueError('Incomplete E39')
    base.check_bindings(stage['bindings'])
    paths = [PROTOCOL,Path(__file__),Path('tests/test_actual_step.py'),base.OUTPUT,verify,e39,
             Path('scripts/train_pair_gradient.py'),Path('scripts/train_frozen_bridge.py')]
    common = {str(p):base.sha256(p) for p in paths}
    bindings = dict(common); data = {r:[] for r in base.ROLES}; traces = []; inventory = None
    for fold in range(5):
        frames, variables, reports, trace = run_fold(fold,parent,common)
        if inventory is not None and inventory != variables: raise ValueError('Changed inventory')
        inventory = variables; traces.append(trace)
        for role in base.ROLES: data[role].append(frames[role])
        bindings.update({str(p):base.sha256(p) for p in reports})
    frames = {r:pd.concat(fs,ignore_index=True) for r,fs in data.items()}
    results = {r:describe(f,inventory) for r,f in frames.items()}
    for i,v in enumerate(inventory):
        f = frames['selection'][frames['selection'].variable == v]
        if len(f): results['selection'][v] = assess(f,base.SEED+i*1000)
    screens = {c:[v for v,r in results['selection'].items() if r['status']=='scored' and r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    summary = dict(screens=screens,full_screen=sorted(set.intersection(*(set(x) for x in screens.values()))),
                   occurrences={r:len(f) for r,f in frames.items()})
    base.check_bindings(bindings)
    base.write_json(OUTPUT,dict(complete=True,bindings=bindings,results=results,summary=summary,
        training=trace_summary(pd.concat(traces,ignore_index=True)), independent_verification=False,
        runtime_modified=False,promotion_allowed=False,
        limitation='Actual-step intervention, reused development current reconstruction, not future efficacy.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()

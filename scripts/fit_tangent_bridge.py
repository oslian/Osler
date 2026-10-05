"""E43: fixed local joint solve, verified in the actual shared recurrent state."""
from copy import deepcopy
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
import torch
from torch.func import functional_call

from scripts import train_actual_step as previous
from scripts.diagnose_parameter_path import clean_example, WEIGHT

base = previous.base
DEST = base.ROOT / 'tangent_bridge_20261005'
OUTPUT = Path('artifacts/native_tangent_bridge_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_TANGENT_BRIDGE_PROTOCOL_20261005.md')
PRIMARY = 'tangent_step'
CONTROLS = ('actual_step', *previous.CONTROLS)
STEPS = tuple(2.**-i for i in range(8))


def huber(residual):
    r = np.asarray(residual, dtype=float)
    return np.where(abs(r) < 1., .5*r*r, abs(r)-.5)


def solve_tangent(jacobian, residual):
    j, r = np.asarray(jacobian, dtype=float), np.asarray(residual, dtype=float)
    if (j.ndim != 2 or not j.size or r.shape != (len(j),)
            or not np.isfinite(j).all() or not np.isfinite(r).all()):
        raise ValueError('Finite aligned Jacobian and residual required')
    kernel = j @ j.T
    eigen, vectors = eigh(kernel, check_finite=True, driver='evd')
    threshold = max(j.shape)*np.finfo(float).eps*max(0., float(eigen[-1]))
    if float(eigen[0]) < -max(threshold, np.finfo(float).tiny):
        raise ValueError('Materially indefinite Gram matrix')
    keep = eigen > threshold
    regularization = .001*float(np.trace(kernel))/len(j)
    report = dict(rows=len(j), parameters=j.shape[1], retained_dimension=int(keep.sum()),
        gram_cutoff=threshold, min_eigenvalue=float(eigen[0]), max_eigenvalue=float(eigen[-1]),
        regularization=regularization, start_objective=float(huber(r).mean()))
    if not keep.any():
        return np.zeros(j.shape[1]), dict(report, converged=True, iterations=0,
            gradient_inf=0., final_objective=report['start_objective'], solver_message='zero Jacobian')
    v, root = vectors[:, keep], np.sqrt(eigen[keep])
    design = v*root

    def objective(q):
        error = design @ q-r
        value = (huber(error).sum()+.5*regularization*np.dot(q, q))/len(r)
        gradient = (design.T @ np.clip(error, -1., 1.)+regularization*q)/len(r)
        return float(value), gradient

    scale = np.sqrt(eigen[keep]+regularization)

    def conditioned(z):
        value, gradient = objective(z/scale)
        return value, gradient/scale

    solution = minimize(conditioned, np.zeros(len(root)), jac=True, method='L-BFGS-B',
        options=dict(maxiter=2000, maxls=50, ftol=1e-12, gtol=1e-8))
    q = solution.x/scale
    loss, gradient = objective(q)
    norm = float(np.max(abs(gradient)))
    if (not solution.success or norm > 1e-5 or not np.isfinite(loss)
            or loss > report['start_objective']+1e-12):
        raise ValueError(f'Unconverged tangent solve: {solution.message}; gradient={norm}')
    weight = j.T @ (v @ (q/root))
    np.testing.assert_allclose(j @ weight, design @ q, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weight @ weight, q @ q, atol=1e-7, rtol=1e-7)
    return weight, dict(report, converged=True, iterations=int(solution.nit),
        gradient_inf=norm, final_objective=loss, solver_message=str(solution.message))


def prediction_gradient(model, example):
    with torch.inference_mode(False), torch.enable_grad():
        e = clean_example(example, model.bridge.projection.weight.dtype)
        weight = torch.zeros_like(model.bridge.projection.weight, requires_grad=True)
        value = functional_call(model, {WEIGHT: weight}, (e,))
        gradient, = torch.autograd.grad(value, weight)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError('Nonfinite prediction Jacobian')
        return float(value.detach()), gradient.detach().double().flatten().numpy().copy()


def point_at(model, example, weight):
    with torch.inference_mode(False), torch.no_grad():
        e = clean_example(example, model.bridge.projection.weight.dtype)
        w = torch.as_tensor(weight, dtype=model.bridge.projection.weight.dtype).reshape_as(
            model.bridge.projection.weight)
        value = float(functional_call(model, {WEIGHT: w}, (e,)))
    if not np.isfinite(value):
        raise ValueError('Nonfinite recurrent prediction')
    return value


def check_derivative(double, e, gradient, direction):
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return 0.
    d = direction/norm
    slope = float(gradient @ d)
    errors = []
    for h in (.001, .0005):
        fd = (point_at(double, e, h*d)-point_at(double, e, -h*d))/(2*h)
        error = abs(fd-slope)
        if error > 2e-5*(1+abs(slope)):
            raise ValueError('Full-Jacobian direction failed finite difference')
        errors.append(error)
    return max(errors)


def fit_step(model, examples, direction, slopes, start, truth, regularization):
    if not (len(examples) == len(slopes) == len(start) == len(truth)):
        raise ValueError('Misaligned fitting line search')
    origin = float(huber(start-truth).mean())
    derivative = float(np.mean(np.clip(start-truth, -1., 1.)*slopes))
    trials = []
    if derivative < 0:
        for alpha in STEPS:
            predictions = np.array([point_at(model, e, alpha*direction) for e in examples])
            penalty = regularization*np.dot(alpha*direction, alpha*direction)/(2*len(truth))
            objective = float(huber(predictions-truth).mean()+penalty)
            accepted = objective <= origin+1e-4*alpha*derivative
            trials.append(dict(alpha=alpha, objective=objective, accepted=bool(accepted)))
            print('E43 fit-only step', trials[-1], flush=True)
            if accepted:
                return alpha, predictions, dict(origin=origin, derivative=derivative, trials=trials)
    return 0., np.asarray(start).copy(), dict(origin=origin, derivative=derivative, trials=trials)


def describe(frame, inventory):
    base.validate_identities(frame)
    names = [c for c in (PRIMARY, 'local_linear', *CONTROLS) if c in frame]
    if PRIMARY not in names or 'bound_start' not in names or not set(frame.variable) <= set(inventory):
        raise ValueError('Invalid comparison inventory')
    if not np.isfinite(frame[['truth', *names]].to_numpy()).all():
        raise ValueError('Nonfinite comparison')
    results = {}
    for v in inventory:
        f = frame[frame.variable == v]
        if f.empty:
            results[v] = dict(status='unsupported'); continue
        link = f.drop_duplicates('patient').set_index('patient').component
        entry = dict(status='scored', patients=f.patient.nunique(), components=f.component.nunique(),
            occurrences=len(f))
        for metric in ('mae', 'smooth_l1'):
            errors = f[names].subtract(f.truth, axis=0)
            values = errors.abs() if metric == 'mae' else pd.DataFrame(huber(errors), index=f.index, columns=names)
            people = values.groupby(f.patient).mean()
            groups = people.groupby(link.reindex(people.index)).mean()
            entry[metric] = dict(patient=people.mean().to_dict(), component=groups.mean().to_dict())
        results[v] = entry
    return results


def assess(frame, seed):
    if frame.variable.nunique() != 1 or not set(CONTROLS) <= set(frame):
        raise ValueError('Incomplete fixed controls')
    result = describe(frame, [frame.variable.iloc[0]])[frame.variable.iloc[0]]
    losses = frame[[PRIMARY, *CONTROLS]].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = losses.groupby(links.reindex(losses.index)).mean()
    result['comparisons'] = {}
    for i, control in enumerate(CONTROLS):
        stats = {level: base.paired_summary(values[PRIMARY]-values[control], seed+i+offset, 5000)
            for level, values, offset in (('patient', losses, 0), ('component', groups, 100))}
        passed = len(losses) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*len(CONTROLS)) for s in stats.values())
        result['comparisons'][control] = dict(paired=stats, screen_pass=bool(passed))
    return result


def scalar_check(frame, inventory):
    result = describe(frame, inventory)
    maximum = 0.
    for v, entry in result.items():
        if entry['status'] != 'scored': continue
        rows = frame[frame.variable == v]
        for metric in ('mae', 'smooth_l1'):
            for name in entry[metric]['patient']:
                people, links = {}, {}
                for r in rows.itertuples():
                    error = abs(getattr(r, name)-r.truth)
                    value = error if metric == 'mae' else (.5*error*error if error < 1 else error-.5)
                    people.setdefault(r.patient, []).append(value); links[r.patient] = r.component
                means = {p: fsum(x)/len(x) for p, x in people.items()}
                groups = {}
                for p, val in means.items(): groups.setdefault(links[p], []).append(val)
                expected = dict(patient=fsum(means.values())/len(means),
                    component=fsum(fsum(x)/len(x) for x in groups.values())/len(groups))
                for level, value in expected.items():
                    difference = abs(value-entry[metric][level][name])
                    if difference > 1e-12*(1+abs(value)):
                        raise ValueError('Scalar aggregation mismatch')
                    maximum = max(maximum, difference)
    return maximum


def run_fold(fold, parent, common):
    variables, examples, bound, original, contract, extraction, _ = previous.prepare_existing(fold)
    olddir = previous.DEST/f'fold{fold}'
    oldcp = olddir/'contract.json'
    if base.sha256(oldcp) != parent['bindings'][str(oldcp)]:
        raise ValueError('Changed E40 contract')
    old = json.loads(oldcp.read_text())
    if old['parent'] != contract or old['extraction'] != extraction or len(variables) != 53:
        raise ValueError('Changed fitting contract')
    frames, sources = {}, [oldcp]
    for role in base.ROLES:
        src = olddir/f'{role}.parquet'
        if base.sha256(src) != parent['bindings'][str(src)]: raise ValueError('Changed E40 rows')
        frame = pd.read_parquet(src); base.validate_identities(frame)
        if len(frame) != (2048 if role == 'fit' else 1024) or not (frame.fold == fold).all():
            raise ValueError('Wrong query scope')
        frames[role] = frame; sources.append(src)
        if role != 'same_source' and list(frame[['patient','component','context','truth']].itertuples(index=False,name=None)) != [
                (e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]:
            raise ValueError('Changed identity/truth')
    if any(set(frames[a].patient)&set(frames[b].patient)
           for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Patient role overlap')
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True, exist_ok=True)
    report = dest/'complete.json'
    bindings = dict(common, **{str(p):base.sha256(p) for p in sources})
    if report.exists():
        completed = json.loads(report.read_text())
        if completed['bindings'] != bindings: raise ValueError('Changed completed fold')
        base.check_bindings(completed['outputs'])
        return {r:pd.read_parquet(dest/f'{r}.parquet') for r in base.ROLES}, variables, report
    if any(dest.iterdir()): raise ValueError('Unverified partial fold; inspect before resuming')
    base.space_required()
    model = base.AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
    double = deepcopy(model).double()
    fit_examples = [base.select_arm(e,b,'binding') for e,b in zip(examples['fit'],bound['fit'])]
    j = np.empty((len(fit_examples), model.bridge.projection.weight.numel()), dtype=np.float64)
    start = np.empty(len(j))
    for i, e in enumerate(fit_examples):
        start[i], j[i] = prediction_gradient(model,e)
        if start[i] != frames['fit'].bound_start.iloc[i]: raise ValueError('Zero fit parity failed')
        if (i+1)%512 == 0: print('E43 fit Jacobian',fold,i+1,len(j),flush=True)
    truth = frames['fit'].truth.to_numpy()
    direction, solver = solve_tangent(j,truth-start)
    direction = direction.astype(np.float32).astype(float)
    slopes = j @ direction
    linear_loss = float(huber(start+slopes-truth).mean())
    alpha, fit_points, line = fit_step(model,fit_examples,direction,slopes,start,truth,solver['regularization'])
    print('E43 fitted',fold,'step',alpha,'linear loss',linear_loss,flush=True)
    fitted = deepcopy(model)
    with torch.no_grad(): fitted.bridge.projection.weight.copy_(torch.tensor(alpha*direction).reshape_as(fitted.bridge.projection.weight))
    base.assert_frozen(fitted, original)
    checkpoint = dest/'tangent_step.pt'
    temp = checkpoint.with_suffix('.tmp'); torch.save(fitted.state_dict(),temp); temp.replace(checkpoint)
    direction_path = dest/'direction.pt'
    temp = direction_path.with_suffix('.tmp')
    torch.save(torch.tensor(direction,dtype=torch.float32),temp); temp.replace(direction_path)
    loaded = deepcopy(model); loaded.load_state_dict(torch.load(checkpoint,weights_only=False),strict=True)
    base.assert_frozen(loaded,original)
    for k,v in loaded.state_dict().items(): torch.testing.assert_close(v,fitted.state_dict()[k],atol=0,rtol=0)
    outputs = {str(p):base.sha256(p) for p in (checkpoint,direction_path)}; verification = {}
    for role, prior in frames.items():
        if role == 'fit': rows = iter(fit_examples)
        elif role == 'same_source': rows = base.same_source_examples(prior, original, variables, fold)
        else: rows = (base.select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
        records, seen = [], set(); fd_error = parity = 0.
        for i,e in enumerate(rows):
            b,g = (start[i],j[i]) if role == 'fit' else prediction_gradient(model,e)
            if b != prior.bound_start.iloc[i]: raise ValueError('Zero role parity failed')
            point = point_at(model,e,alpha*direction)
            if role == 'fit' and point != fit_points[i]: raise ValueError('Line search replay failed')
            with torch.inference_mode(False), torch.no_grad():
                clean = clean_example(e,torch.float32)
                direct = base.direct_prediction(loaded,clean,clean['organs'])
            error = abs(point-direct); parity = max(parity,error)
            if error > 1e-7: raise ValueError('Independent checkpoint replay failed')
            slope = float(g @ direction)
            if e['target'] not in seen:
                fd_error = max(fd_error,check_derivative(double,e,g,direction)); seen.add(e['target'])
            records.append(dict(tangent_base=b,tangent_slope=slope,local_linear=b+alpha*slope,tangent_step=point))
            if (i+1)%512 == 0: print('E43 replay',fold,role,i+1,len(prior),flush=True)
        if len(records) != len(prior): raise ValueError('Incomplete replay')
        frame = pd.concat([prior.reset_index(drop=True),pd.DataFrame(records)],axis=1)
        scalar = scalar_check(frame,variables)
        verification[role] = dict(rows=len(frame),checkpoint_replay_max_error=parity,
            fd_checks=len(seen),fd_max_error=fd_error,scalar_max_error=scalar)
        path = dest/f'{role}.parquet'; base.space_required()
        tmp = path.with_suffix('.tmp'); frame.to_parquet(tmp,index=False); tmp.replace(path)
        outputs[str(path)] = base.sha256(path); frames[role] = frame
    base.assert_frozen(loaded,original); base.check_bindings(bindings)
    base.write_json(report,dict(complete=True,bindings=bindings,outputs=outputs,solver=solver,
        linear_fit_loss=linear_loss,accepted_alpha=alpha,line_search=line,verification=verification,
        runtime_modified=False,promotion_allowed=False))
    print('E43 fold complete',fold,flush=True)
    return frames, variables, report


def main():
    if OUTPUT.exists(): raise ValueError('Completed experiment exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    verify = Path('artifacts/native_actual_step_verification_20261005.json')
    parent, cert = json.loads(previous.OUTPUT.read_text()), json.loads(verify.read_text())
    if not parent['complete'] or not cert['complete'] or cert['bindings'][str(previous.OUTPUT)] != base.sha256(previous.OUTPUT):
        raise ValueError('Verified E40 required')
    base.check_bindings(parent['bindings']); base.check_bindings(cert['bindings'])
    paths = [PROTOCOL,Path('RECONSTRUCTION_TANGENT_NUMERICS_20261005.md'),
        DEST/'failed_initial/fit_tangent_bridge.py',Path(__file__),Path('tests/test_tangent_bridge.py'),previous.OUTPUT,verify,
        Path('scripts/diagnose_parameter_path.py'),Path('scripts/train_actual_step.py'),
        Path('scripts/train_frozen_bridge.py'),Path('scripts/train_organ_assimilation.py'),
        Path('scripts/organ_assimilation.py'),Path('scripts/train_organ_binding.py'),
        Path('scripts/audit_trained_bridge.py'),Path('whole_body_jepa/physiology_state.py')]
    common = {str(p):base.sha256(p) for p in paths}
    bindings = dict(common); parts = {r:[] for r in base.ROLES}; training = []
    inventory = None
    for fold in range(5):
        frames,variables,report = run_fold(fold,parent,common)
        if inventory is not None and inventory != variables: raise ValueError('Changed target inventory')
        inventory = variables
        info = json.loads(report.read_text())
        training.append({k:v for k,v in info.items() if k not in ('bindings','outputs')})
        bindings[str(report)] = base.sha256(report); bindings.update(info['outputs'])
        for r in base.ROLES: parts[r].append(frames[r])
    frames = {r:pd.concat(fs,ignore_index=True) for r,fs in parts.items()}
    results = {r:describe(f,inventory) for r,f in frames.items()}
    scalar = {r:scalar_check(f,inventory) for r,f in frames.items()}
    for i,v in enumerate(inventory):
        f = frames['selection'][frames['selection'].variable == v]
        if len(f): results['selection'][v] = assess(f,base.SEED+i*1000)
    screens = {c:[v for v,r in results['selection'].items() if r['status']=='scored' and r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    summary = dict(occurrences={r:len(f) for r,f in frames.items()},screens=screens,
        full_screen=sorted(set.intersection(*(set(v) for v in screens.values()))),scalar_max_errors=scalar)
    base.check_bindings(bindings)
    base.write_json(OUTPUT,dict(complete=True,bindings=bindings,results=results,summary=summary,
        training=training,runtime_modified=False,promotion_allowed=False,
        limitation='Local linearized fit and nonlinear intervention on reused development, not an optimizer causal study or clinical validation.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()

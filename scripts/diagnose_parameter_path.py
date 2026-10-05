"""E36: recurrent saved-parameter path versus final-output chord."""
from collections import defaultdict
from copy import deepcopy
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call

from scripts.audit_trained_bridge import check_bindings, same_source_examples
from scripts.diagnose_bridge_geometry import geometry, summarize as output_summary
from scripts.diagnose_organ_assimilation import require_verification
from scripts.train_frozen_bridge import DEST as PARENT_DIR, OUTPUT as PARENT, ROLES, assert_frozen
from scripts.train_organ_assimilation import AssimilatedReconstruction, validate_identities
from scripts.train_organ_binding import prepare, select_arm, space_required, write_json
from scripts.train_content_retention import DEST as CONTENT_DIR
from whole_body_jepa.multisource_intervals import sha256

ROOT = PARENT_DIR.parent
DEST = ROOT / 'parameter_path_20261005'
OUTPUT = Path('artifacts/native_parameter_path_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_PARAMETER_PATH_PROTOCOL_20261005.md')
VERIFICATION = Path('artifacts/native_frozen_bridge_verification_20261005.json')
WEIGHT = 'bridge.projection.weight'


def clean_example(example, dtype):
    # The same-source generator yields inside inference_mode. New ordinary
    # tensors are necessary for gradients even when weights are frozen.
    return {k: v.detach().to(dtype=dtype).clone() if k in ('body','organs','presence')
            else v.detach().clone() if isinstance(v,torch.Tensor) else v
            for k,v in example.items()}


def path_value(model, example, alpha):
    weight = model.bridge.projection.weight
    return functional_call(model, {WEIGHT: weight * alpha}, (example,))


def tangent(model, example):
    with torch.inference_mode(False), torch.enable_grad():
        e = clean_example(example, model.bridge.projection.weight.dtype)
        alpha = model.bridge.projection.weight.new_tensor(0., requires_grad=True)
        value = path_value(model, e, alpha)
        slope, = torch.autograd.grad(value, alpha)
        result = float(value.detach()), float(slope.detach())
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite parameter derivative')
        return result


def endpoints(model, example):
    with torch.inference_mode(False), torch.no_grad():
        e = clean_example(example, model.bridge.projection.weight.dtype)
        values = [float(path_value(model, e, a)) for a in (0., 1.)]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite endpoints')
        return values


def finite_difference(model, example, slope):
    with torch.inference_mode(False), torch.no_grad():
        e = clean_example(example, torch.float64)
        errors = []
        for h in (.001, .0005):
            fd = float((path_value(model, e, h)-path_value(model, e, -h))/(2*h))
            error = abs(fd-slope)
            if not np.isfinite(fd) or error > 1e-6*(1+abs(slope)):
                raise ValueError('Autograd/finite difference mismatch')
            errors.append(error)
        return max(errors)


def path_metrics(base, prediction, truth, slope):
    rows = geometry(base, prediction, truth)
    r = np.asarray(truth, dtype=float)-np.asarray(base, dtype=float)
    s = np.asarray(slope, dtype=float)
    if s.shape != r.shape or not np.isfinite(s).all():
        raise ValueError('Invalid path slope')
    d = np.where(r == 0, abs(s), -np.sign(r)*s)
    rows['parameter_derivative'] = d
    rows['nonlinear_remainder'] = rows.mae_right_derivative-d
    rows['parameter_smooth_derivative'] = -np.clip(r, -1., 1.)*s
    rows['slope_chord_gap'] = abs(s-(np.asarray(prediction)-np.asarray(base)))
    if not np.isfinite(rows.to_numpy()).all():
        raise ValueError('Nonfinite path metric')
    np.testing.assert_allclose(rows.mae_delta,
        rows.parameter_derivative+rows.nonlinear_remainder+rows.crossing_curvature,
        atol=1e-12, rtol=1e-12)
    return rows


def summarize(frame, inventory):
    validate_identities(frame)
    reference = frame.assign(bound_start=frame.base64, fixed_immediate=frame.pred64)
    original, maximum = output_summary(reference, inventory, 'fixed_immediate')
    path_metrics(frame.base64, frame.pred64, frame.truth, frame.slope64)
    keys = ['parameter_derivative', 'nonlinear_remainder',
            'parameter_smooth_derivative', 'slope_chord_gap']
    for target, result in original.items():
        if result['status'] != 'scored': continue
        f = frame[frame.variable == target]
        metrics = path_metrics(f.base64, f.pred64, f.truth, f.slope64)[keys]
        metrics['patient'] = f.patient.to_numpy()
        people = metrics.groupby('patient')[keys].mean()
        links = f.drop_duplicates('patient').set_index('patient').component
        groups = people.groupby(links.reindex(people.index)).mean()
        levels = dict(occurrence=metrics[keys].mean().to_dict(),
                      patient=people.mean().to_dict(), component=groups.mean().to_dict())
        scalar_people, scalar_groups, all_rows = defaultdict(list), defaultdict(list), []
        for row in f.itertuples():
            r, c, s = row.truth-row.base64, row.pred64-row.base64, row.slope64
            d = -s if r > 0 else s if r < 0 else abs(s)
            g = -c if r > 0 else c if r < 0 else abs(c)
            smooth = -s if r > 1 else s if r < -1 else -r*s
            v = dict(zip(keys, (d, g-d, smooth, abs(s-c))))
            scalar_people[row.patient].append(v); all_rows.append(v)
        def mean(values):
            return {k: fsum(x[k] for x in values)/len(values) for k in keys}
        scalar_people = {p: mean(v) for p, v in scalar_people.items()}
        for p, v in scalar_people.items(): scalar_groups[links[p]].append(v)
        scalar = dict(occurrence=mean(all_rows), patient=mean(list(scalar_people.values())),
                      component=mean([mean(v) for v in scalar_groups.values()]))
        for level in levels:
            for k, value in levels[level].items():
                error = abs(value-scalar[level][k]); maximum = max(maximum, error)
                if error > 1e-12*(1+abs(value)): raise ValueError('Scalar aggregation mismatch')
            result['metrics'][level].update(levels[level])
    return original, maximum


def classify(results, level):
    cases = dict(full_harm=[], harmful_initial_direction=[], beneficial_initial_direction=[],
        ambiguous_initial_direction=[], harmful_chord_beneficial_initial=[])
    rows = {t: r['metrics'][level] for t, r in results.items() if r['status']=='scored'}
    for t, m in rows.items():
        tol = 1e-10*(1+m['base_mae']+m['prediction_mae'])
        if m['mae_delta'] <= tol: continue
        cases['full_harm'].append(t)
        d = m['parameter_derivative']
        name = ('harmful_initial_direction' if d > tol else
                'beneficial_initial_direction' if d < -tol else 'ambiguous_initial_direction')
        cases[name].append(t)
        if d < -tol and m['mae_right_derivative'] > tol:
            cases['harmful_chord_beneficial_initial'].append(t)
    return dict(supported=len(rows), counts={k:len(v) for k,v in cases.items()}, targets=cases,
        macro={k:fsum(m[k] for m in rows.values())/len(rows) for k in next(iter(rows.values()), {})})


def run_fold(fold, parent, common):
    # Refuse to build new parent caches during a frozen diagnostic.
    for role in ('fit', 'selection'):
        for path in (ROOT/'organ_binding_20261004'/f'fold{fold}'/(role+'_binding.joblib'),
                     CONTENT_DIR/f'fold{fold}'/(role+'.joblib')):
            if not path.exists() or not path.with_suffix('.json').exists():
                raise ValueError('Missing existing parent cache')
    variables, examples, bound, original, contract, extraction, _ = prepare(fold)
    mp = Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
    if sha256(mp) != parent['bindings'][str(mp)]: raise ValueError('Changed parent fold')
    meta = json.loads(mp.read_text())
    if meta['contract']['parent'] != contract or meta['contract']['extraction'] != extraction:
        raise ValueError('Changed input contract')
    check_bindings(meta['contract']['bindings']); check_bindings(meta['outputs'])
    checkpoint = PARENT_DIR/f'fold{fold}'/'immediate.pt'
    if sha256(checkpoint) != meta['fits']['immediate']['sha256']:
        raise ValueError('Changed checkpoint')
    model = AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
    saved = torch.load(checkpoint, weights_only=False); model.load_state_dict(saved, strict=True)
    assert_frozen(model, original)
    double = deepcopy(model).double().eval().requires_grad_(False)
    bindings = dict(common, **{str(p):sha256(p) for p in (mp, checkpoint)})
    frames = {r:pd.read_parquet(PARENT_DIR/f'fold{fold}'/f'{r}.parquet') for r in ROLES}
    for r, f in frames.items():
        validate_identities(f)
        if len(f) != (2048 if r=='fit' else 1024) or not (f.fold==fold).all():
            raise ValueError('Changed role/fold')
    if any(set(frames[a].patient)&set(frames[b].patient)
           for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Role overlap')
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for role, frame in frames.items():
        path, report = dest/f'{role}.parquet', dest/f'{role}.json'
        role_bindings = dict(bindings, **{str(PARENT_DIR/f'fold{fold}'/f'{role}.parquet'):
                                         meta['outputs'][str(PARENT_DIR/f'fold{fold}'/f'{role}.parquet')]})
        if report.exists():
            info = json.loads(report.read_text())
            if info['bindings'] != role_bindings or info['sha256'] != sha256(path):
                raise ValueError('Changed completed path audit')
            outputs[role] = (pd.read_parquet(path), info, report); continue
        if path.exists(): raise ValueError('Unverified partial output')
        if role == 'same_source': rows = same_source_examples(frame, original, variables, fold)
        else:
            expected = [(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
            if list(frame[['patient','component','context','truth']].itertuples(index=False,name=None)) != expected:
                raise ValueError('Mismatched query identities')
            rows = (select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
        records, seen = [], set()
        max_precision = max_fd = max_parity = 0.
        for i, e in enumerate(rows):
            base32, pred32 = endpoints(model, e)
            old = frame.iloc[i]
            if (base32, pred32) != (old.bound_start, old.fixed_immediate):
                raise ValueError('Float32 saved endpoint parity failed')
            base64, slope = tangent(double, e)
            double_zero, pred64 = endpoints(double, e)
            if double_zero != base64: raise ValueError('Grad/no-grad zero mismatch')
            for p, ref in ((base64, base32),(pred64,pred32)):
                error = abs(p-ref); max_precision = max(max_precision,error)
                if error > 2e-5*(1+abs(ref)): raise ValueError('Excess precision drift')
            checked = e['target'] not in seen
            if checked:
                max_fd = max(max_fd, finite_difference(double,e,slope)); seen.add(e['target'])
            records.append(dict(base64=base64,pred64=pred64,slope64=slope,fd_checked=checked))
            if (i+1)%512 == 0: print('E36 replay',fold,role,i+1,len(frame),flush=True)
        if len(records) != len(frame): raise ValueError('Incomplete path replay')
        result = pd.concat([frame.reset_index(drop=True),pd.DataFrame(records)],axis=1)
        path_metrics(result.base64,result.pred64,result.truth,result.slope64)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        check_bindings(role_bindings); space_required()
        temp = path.with_suffix('.tmp'); result.to_parquet(temp,index=False); temp.replace(path)
        info = dict(bindings=role_bindings,sha256=sha256(path),rows=len(result),
            fd_checks=len(seen),max_fd_error=max_fd,max_precision_error=max_precision,
            max_float32_parity_error=max_parity,weights_unchanged=True)
        write_json(report,info); outputs[role]=(result,info,report)
        print('E36 role complete',fold,role,len(result),'FD',len(seen),flush=True)
    return outputs, variables


def main():
    if OUTPUT.exists(): raise ValueError('Completed diagnostic exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    parent=json.loads(PARENT.read_text()); require_verification(PARENT,json.loads(VERIFICATION.read_text()))
    check_bindings(parent['bindings'])
    dependencies = [PARENT,VERIFICATION,PROTOCOL,Path(__file__),Path('tests/test_parameter_path.py'),
        Path('scripts/diagnose_bridge_geometry.py'),Path('scripts/train_frozen_bridge.py'),
        Path('scripts/train_organ_assimilation.py'),Path('scripts/train_organ_binding.py'),
        Path('scripts/organ_assimilation.py'),Path('scripts/audit_trained_bridge.py'),
        Path('whole_body_jepa/physiology_state.py')]
    common={str(p):sha256(p) for p in dependencies}
    bindings=dict(common); parts={r:[] for r in ROLES}; replay={r:[] for r in ROLES}
    inventory=list(parent['results']['fit'])
    if len(inventory)!=53: raise ValueError('Wrong inventory')
    for fold in range(5):
        outputs, variables=run_fold(fold,parent,common)
        if set(variables)!=set(inventory): raise ValueError('Changed inventory')
        for role,(frame,info,path) in outputs.items():
            parts[role].append(frame); replay[role].append({k:v for k,v in info.items() if k!='bindings'})
            bindings[str(path)]=sha256(path)
            private=path.with_suffix('.parquet'); bindings[str(private)]=sha256(private)
    results, summary, maximum={}, {}, 0.
    for role in ROLES:
        frame=pd.concat(parts[role],ignore_index=True)
        results[role],err=summarize(frame,inventory); maximum=max(maximum,err)
        summary[role]={l:classify(results[role],l) for l in ('occurrence','patient','component')}
        print('E36',role,summary[role]['patient']['counts'],flush=True)
    check_bindings(bindings)
    write_json(OUTPUT,dict(inventory=inventory,bindings=bindings,replay=replay,results=results,
        summary=summary,maximum_scalar_roundoff=maximum,complete=True,
        runtime_modified=False,promotion_allowed=False,
        limitation='Immediate saved parameter direction; local diagnostic on reused current reconstruction, '
                   'not a convex-path guarantee, training trajectory, future forecast or causal/clinical repair.'))


if __name__=='__main__': main()

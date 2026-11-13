"""Fixed-weight E43 bridge decomposition; no refit or deployment selection."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import svd
import torch

from scripts import fit_tangent_bridge as parent

base = parent.base
DEST = base.ROOT / 'tangent_spectrum_20261005'
OUTPUT = Path('artifacts/native_tangent_spectrum_20261005.json')
PROTOCOL = Path('RECONSTRUCTION_TANGENT_SPECTRUM_PROTOCOL_20261005.md')
VARIANTS = ('strong_actual', 'weak_actual', 'strong_linear', 'weak_linear')


def split_direction(jacobian, direction, regularization):
    j, w = np.asarray(jacobian, dtype=float), np.asarray(direction, dtype=float)
    if (j.ndim != 2 or not j.size or w.shape != (j.shape[1],)
            or not np.isfinite(j).all() or not np.isfinite(w).all()
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError('Finite aligned Jacobian, direction and lambda required')
    _, singular, right = svd(j, full_matrices=False, lapack_driver='gesdd')
    eigen, right = (singular**2)[::-1], right[::-1]
    cutoff = max(j.shape)*np.finfo(float).eps*max(0., float(eigen[-1]))
    if eigen[0] < -max(cutoff, np.finfo(float).tiny):
        raise ValueError('Indefinite Gram matrix')
    retained = eigen > cutoff
    strong = retained & (eigen >= regularization)
    weak = retained & ~strong
    jw = j @ w

    def project(mask):
        v = right[mask]
        return v.T @ (v @ w) if mask.any() else np.zeros_like(w)

    parts = dict(strong=project(strong), weak=project(weak))
    parts['null'] = w-parts['strong']-parts['weak']
    energies = {k:float(x @ x) for k,x in parts.items()}
    norm2 = float(w @ w)
    orthogonality = max(abs(float(parts[a] @ parts[b])) /
        (1+np.linalg.norm(parts[a])*np.linalg.norm(parts[b]))
        for a,b in (('strong','weak'),('strong','null'),('weak','null')))
    if orthogonality > 1e-7:
        raise ValueError('Nonorthogonal parameter decomposition')
    np.testing.assert_allclose(sum(energies.values()), norm2, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(j @ parts['null'], 0., atol=1e-7*(1+np.linalg.norm(jw)), rtol=0)
    return parts, dict(retained=int(retained.sum()), strong=int(strong.sum()),
        weak=int(weak.sum()), cutoff=cutoff, min_eigenvalue=float(eigen[0]),
        max_eigenvalue=float(eigen[-1]), regularization=float(regularization),
        direction_norm2=norm2, energy=energies,
        energy_fraction={k:(v/norm2 if norm2 else 0.) for k,v in energies.items()},
        orthogonality_error=float(orthogonality),
        fit_delta_energy={k:float(np.mean((j @ x)**2)) for k,x in parts.items()})


def role_rows(role, frame, examples, bound, original, variables, fold):
    if role == 'same_source':
        return base.same_source_examples(frame, original, variables, fold)
    identities = [(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
    if identities != list(frame[['patient','component','context','truth']].itertuples(index=False,name=None)):
        raise ValueError('Changed original query identity')
    return (base.select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))


def cast_parts(parts, direction):
    if set(parts) != {'strong', 'weak', 'null'}:
        raise ValueError('Expected three orthogonal parts')
    direction = np.asarray(direction, dtype=float)
    if not np.isfinite(direction).all() or any(
            np.asarray(v).shape != direction.shape or not np.isfinite(v).all()
            for v in parts.values()):
        raise ValueError('Invalid finite parameter parts')
    np.testing.assert_allclose(sum(parts.values()), direction, atol=1e-12, rtol=1e-12)
    cast = {k:v.astype(np.float32).astype(float) for k,v in parts.items()}
    # Independent casting is not additive; keep its residual out of model weights.
    cast['roundoff'] = direction-sum(cast.values())
    return cast


def descriptions(frame, inventory):
    if not np.isfinite(frame[list(VARIANTS)].to_numpy()).all():
        raise ValueError('Invalid ablation predictions')
    output = {}
    maximum = 0.
    # Reuse the exact original weighting and independent scalar implementation.
    # "tangent_step" inside each variant is that ablation, not a new trained model.
    for variant in VARIANTS:
        view = frame.assign(tangent_step=frame[variant])
        output[variant] = parent.describe(view, inventory)
        maximum = max(maximum, parent.scalar_check(view, inventory))
    return output, maximum


def run_fold(fold, report, common):
    variables, examples, bound, original, contract, extraction, _ = parent.previous.prepare_existing(fold)
    src = parent.DEST/f'fold{fold}'
    info_path = src/'complete.json'
    if base.sha256(info_path) != report['bindings'][str(info_path)]:
        raise ValueError('Changed original completion record')
    info = json.loads(info_path.read_text())
    base.check_bindings(info['bindings']); base.check_bindings(info['outputs'])
    frames = {role:pd.read_parquet(src/f'{role}.parquet') for role in base.ROLES}
    for role,f in frames.items():
        base.validate_identities(f)
        if len(f) != (2048 if role=='fit' else 1024) or not (f.fold==fold).all():
            raise ValueError('Changed role size')
    if any(set(frames[a].patient)&set(frames[b].patient) for a,b in
           (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Overlapping patient roles')
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    completion = dest/'complete.json'
    bindings = dict(common, **{str(info_path):base.sha256(info_path)}, **info['outputs'])
    if completion.exists():
        done = json.loads(completion.read_text())
        if done['bindings'] != bindings: raise ValueError('Changed completed diagnostic')
        base.check_bindings(done['outputs'])
        return {r:pd.read_parquet(dest/f'{r}.parquet') for r in base.ROLES},variables,completion
    if any(dest.iterdir()): raise ValueError('Unverified partial output; inspect before resuming')
    base.space_required()
    model = base.AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
    direction = torch.load(src/'direction.pt',weights_only=False).double().numpy()
    alpha = info['accepted_alpha']
    fit_examples = list(role_rows('fit',frames['fit'],examples,bound,original,variables,fold))
    j = np.empty((len(fit_examples),len(direction)))
    for i,e in enumerate(fit_examples):
        start,j[i] = parent.prediction_gradient(model,e)
        if start != frames['fit'].tangent_base.iloc[i]: raise ValueError('Changed zero prediction')
        if (i+1)%512==0: print('E44 Jacobian',fold,i+1,len(j),flush=True)
    np.testing.assert_allclose(j @ direction,frames['fit'].tangent_slope,rtol=1e-7,atol=1e-7)
    parts,spectrum = split_direction(j,direction,info['solver']['regularization'])
    if spectrum['retained'] != info['solver']['retained_dimension']:
        raise ValueError('Changed retained dimension')
    np.testing.assert_allclose(spectrum['max_eigenvalue'],info['solver']['max_eigenvalue'],rtol=1e-12,atol=1e-12)
    cast = cast_parts(parts,direction)
    spectrum['cast_roundoff_norm2'] = float(cast['roundoff'] @ cast['roundoff'])
    candidates = {}
    for k in ('strong','weak','full'):
        m = deepcopy(model)
        w = alpha*(direction if k=='full' else cast[k])
        with torch.no_grad():m.bridge.projection.weight.copy_(torch.tensor(w).reshape_as(m.bridge.projection.weight))
        base.assert_frozen(m,original);candidates[k]=m
    outputs,verification = {},{}
    for role,prior in frames.items():
        rows = iter(fit_examples) if role=='fit' else role_rows(role,prior,examples,bound,original,variables,fold)
        records=[];parity=cast_error=closure_error=0.
        for i,e in enumerate(rows):
            b,g = (prior.tangent_base.iloc[i],j[i]) if role=='fit' else parent.prediction_gradient(model,e)
            if b != prior.tangent_base.iloc[i]:raise ValueError('Changed query start')
            total=float(g @ direction)
            np.testing.assert_allclose(total,prior.tangent_slope.iloc[i],rtol=1e-7,atol=1e-7)
            slopes={k:float(g @ w) for k,w in cast.items()}
            err=abs(sum(slopes.values())-total)
            if err>1e-7*(1+abs(total)):raise ValueError('Cast decomposition mismatch')
            closure_error=max(closure_error,err)
            cast_error=max(cast_error,abs(slopes['roundoff']))
            rec={f'{k}_slope':v for k,v in slopes.items()}
            with torch.no_grad():
                for k,m in candidates.items():
                    p=base.direct_prediction(m,e,e['organs'])
                    if k=='full':
                        if p!=prior.tangent_step.iloc[i]:raise ValueError('Changed full model')
                    else:
                        functional=parent.point_at(model,e,alpha*cast[k])
                        error=abs(p-functional);parity=max(parity,error)
                        if error>1e-7:raise ValueError('Independent ablation replay mismatch')
                        rec[f'{k}_actual']=p
                        rec[f'{k}_linear']=b+alpha*slopes[k]
            records.append(rec)
            if (i+1)%512==0:print('E44 replay',fold,role,i+1,len(prior),flush=True)
        if len(records)!=len(prior):raise ValueError('Incomplete role replay')
        f=pd.concat([prior.reset_index(drop=True),pd.DataFrame(records)],axis=1)
        _,scalar=descriptions(f,variables)
        verification[role]=dict(rows=len(f),actual_replay_max_error=parity,
            cast_roundoff_slope_max=cast_error,complete_decomposition_max_error=closure_error,
            scalar_max_error=scalar)
        path=dest/f'{role}.parquet';base.space_required()
        temp=path.with_suffix('.tmp');f.to_parquet(temp,index=False);temp.replace(path)
        outputs[str(path)]=base.sha256(path);frames[role]=f
    p=dest/'parts.pt';torch.save({k:torch.tensor(v) for k,v in cast.items()},p)
    outputs[str(p)]=base.sha256(p)
    base.check_bindings(bindings)
    base.write_json(completion,dict(complete=True,fold=fold,alpha=alpha,spectrum=spectrum,
        verification=verification,bindings=bindings,outputs=outputs))
    print('E44 fold complete',fold,spectrum['strong'],spectrum['weak'],flush=True)
    return frames,variables,completion


def main():
    if OUTPUT.exists():raise ValueError('Completed diagnostic exists')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    cert_path=Path('artifacts/native_tangent_bridge_verification_20261005.json')
    report=json.loads(parent.OUTPUT.read_text());cert=json.loads(cert_path.read_text())
    if not report['complete'] or not cert['complete'] or cert['bindings'][str(parent.OUTPUT)]!=base.sha256(parent.OUTPUT):
        raise ValueError('Verified E43 required')
    base.check_bindings(report['bindings']);base.check_bindings(cert['bindings'])
    numerical_path=Path('artifacts/native_tangent_projection_numerics_20261005.json')
    numerical=json.loads(numerical_path.read_text())
    for name,digest in numerical['bindings'].items():
        path=(DEST/'failed_gram_projection'/'audit_tangent_spectrum.py'
              if name=='scripts/audit_tangent_spectrum.py' else Path(name))
        if base.sha256(path)!=digest:raise ValueError('Changed numerical diagnosis snapshot')
    paths=[Path(__file__),PROTOCOL,Path('tests/test_tangent_spectrum.py'),parent.OUTPUT,
        cert_path,Path('scripts/fit_tangent_bridge.py'),
        Path('RECONSTRUCTION_TANGENT_SPECTRUM_NUMERICS_20261005.md'),numerical_path,
        *(DEST/'failed_gram_projection'/name for name in ('audit_tangent_spectrum.py',
            'test_tangent_spectrum.py','RECONSTRUCTION_TANGENT_SPECTRUM_NUMERICS_20261005.md')),
        *(DEST/'failed_initial'/name for name in ('audit_tangent_spectrum.py',
            'test_tangent_spectrum.py','RECONSTRUCTION_TANGENT_SPECTRUM_PROTOCOL_20261005.md'))]
    common={str(p):base.sha256(p) for p in paths};bindings=dict(common)
    data={r:[] for r in base.ROLES};details=[];inventory=None
    for fold in range(5):
        frames,variables,completion=run_fold(fold,report,common)
        if inventory is not None and inventory!=variables:raise ValueError('Changed inventory')
        inventory=variables;info=json.loads(completion.read_text())
        details.append({k:v for k,v in info.items() if k not in ('bindings','outputs')})
        bindings[str(completion)]=base.sha256(completion);bindings.update(info['outputs'])
        for r in data:data[r].append(frames[r])
    results={};errors={}
    for role,parts in data.items():
        results[role],errors[role]=descriptions(pd.concat(parts,ignore_index=True),inventory)
    base.check_bindings(bindings)
    base.write_json(OUTPUT,dict(complete=True,bindings=bindings,details=details,
        results=results,scalar_max_errors=errors,runtime_modified=False,promotion_allowed=False))
    print('E44 complete',errors,flush=True)


if __name__=='__main__':main()

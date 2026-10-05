"""Fit-only numerical diagnosis of E44 Gram back-projection; no query scoring."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh, svd
import torch

from scripts import audit_tangent_spectrum as audit

OUTPUT = Path('artifacts/native_tangent_projection_numerics_20261005.json')


def diagnostics(j, w, parts):
    return dict(orthogonality=max(abs(float(parts[a] @ parts[b])) /
        (1+np.linalg.norm(parts[a])*np.linalg.norm(parts[b]))
        for a,b in (('strong','weak'),('strong','null'),('weak','null'))),
        energy_error=float(abs(sum(x@x for x in parts.values())-w@w)),
        null_response_max=float(np.max(abs(j@parts['null']))),
        reconstruction_error=float(np.max(abs(sum(parts.values())-w))))


def main():
    if OUTPUT.exists():raise ValueError('Existing numerical evidence')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    fold=1
    variables,examples,bound,original,*_=audit.parent.previous.prepare_existing(fold)
    src=audit.parent.DEST/f'fold{fold}'
    info=json.loads((src/'complete.json').read_text())
    audit.base.check_bindings(info['bindings']);audit.base.check_bindings(info['outputs'])
    frame=pd.read_parquet(src/'fit.parquet')
    rows=list(audit.role_rows('fit',frame,examples,bound,original,variables,fold))
    model=audit.base.AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
    w=torch.load(src/'direction.pt',weights_only=True).double().numpy()
    j=np.empty((len(rows),len(w)))
    for i,e in enumerate(rows):
        b,j[i]=audit.parent.prediction_gradient(model,e)
        if b!=frame.tangent_base.iloc[i]:raise ValueError('Changed fit origin')
        if (i+1)%512==0:print('projection numerical Jacobian',i+1,flush=True)
    np.testing.assert_allclose(j@w,frame.tangent_slope,rtol=1e-7,atol=1e-7)
    eigen,left=eigh(j@j.T,driver='evd')
    lam=info['solver']['regularization'];cutoff=max(j.shape)*np.finfo(float).eps*max(0.,eigen[-1])
    masks=dict(strong=(eigen>cutoff)&(eigen>=lam),weak=(eigen>cutoff)&(eigen<lam))
    gram={name:j.T@(left[:,m]@((left[:,m].T@(j@w))/eigen[m])) for name,m in masks.items()}
    gram['null']=w-gram['strong']-gram['weak']
    print('Gram diagnostic',diagnostics(j,w,gram),flush=True)
    _,singular,right=svd(j,full_matrices=False,lapack_driver='gesdd')
    squared=singular**2
    stable_masks=dict(strong=(squared>cutoff)&(squared>=lam),weak=(squared>cutoff)&(squared<lam))
    stable={name:right[m].T@(right[m]@w) for name,m in stable_masks.items()}
    stable['null']=w-stable['strong']-stable['weak']
    counts={k:int(m.sum()) for k,m in masks.items()}
    stable_counts={k:int(m.sum()) for k,m in stable_masks.items()}
    if counts!=stable_counts:raise ValueError('Numerical decomposition changes subspace membership')
    paths=[Path(__file__),Path('scripts/audit_tangent_spectrum.py'),src/'complete.json',src/'direction.pt']
    result=dict(complete=True,fold=fold,fit_rows=len(rows),regularization=lam,cutoff=float(cutoff),
        gram=diagnostics(j,w,gram),direct_svd=diagnostics(j,w,stable),counts=counts,
        eigenvalue_max_difference=float(np.max(abs(eigen-squared[::-1]))),
        parameter_difference_max={k:float(np.max(abs(gram[k]-stable[k]))) for k in gram},
        fit_response_difference_max={k:float(np.max(abs(j@(gram[k]-stable[k])))) for k in gram},
        query_scored=False,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):audit.base.sha256(p) for p in paths})
    audit.base.write_json(OUTPUT,result)
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'},indent=2),flush=True)


if __name__=='__main__':main()

"""E37: momentum-aware gradient origins for an exactly reproduced E34 fit."""
from collections import defaultdict
from copy import deepcopy
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from torch.nn import functional as F

from scripts.audit_trained_bridge import check_bindings, same_source_examples
from scripts.diagnose_parameter_path import (ROOT, DEST as PATH_DIR, OUTPUT as PATH_REPORT,
    PARENT_DIR, PARENT, VERIFICATION, WEIGHT, clean_example)
from scripts.diagnose_organ_assimilation import require_verification
from scripts.train_frozen_bridge import ROLES, assert_frozen
from scripts.train_organ_assimilation import AssimilatedReconstruction, SEED, validate_identities
from scripts.train_organ_binding import prepare, select_arm, space_required, write_json
from whole_body_jepa.multisource_intervals import sha256

DEST=ROOT/'adam_origins_20261005'
OUTPUT=Path('artifacts/native_adam_origins_20261005.json')
PROTOCOL=Path('RECONSTRUCTION_ADAM_ORIGIN_PROTOCOL_20261005.md')


class AdamOrigins:
    """Account for actual updates; never change optimizer or model tensors."""
    def __init__(self, size, targets):
        if size<1 or targets<1: raise ValueError('Invalid attribution dimensions')
        self.moment=np.zeros((targets,size),dtype=float)
        self.parts=np.zeros((targets+2,size),dtype=float)
        self.steps=0; self.max_moment_error=0.

    def record(self,before,after,gradient,state,group,target):
        arrays=[np.asarray(x,dtype=float).reshape(-1) for x in (before,after,gradient)]
        if (not 0<=target<len(self.moment) or any(x.shape!=(self.moment.shape[1],)
                or not np.isfinite(x).all() for x in arrays)):
            raise ValueError('Invalid update')
        before,after,g=arrays
        if any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable')):
            raise ValueError('Unsupported optimizer variant')
        b1,b2=group['betas'];t=int(state['step'].item())
        if t!=self.steps+1: raise ValueError('Nonsequential optimizer step')
        self.steps=t
        self.moment*=b1;self.moment[target]+=(1-b1)*g
        v=state['exp_avg_sq'].detach().double().numpy().reshape(-1)
        m=state['exp_avg'].detach().double().numpy().reshape(-1)
        if not np.isfinite(v).all() or not np.isfinite(m).all() or (v<0).any():
            raise ValueError('Invalid optimizer moments')
        self.max_moment_error=max(self.max_moment_error,float(abs(self.moment.sum(0)-m).max()))
        denom=np.sqrt(v/(1-b2**t))+group['eps']
        updates=-group['lr']/(1-b1**t)*self.moment/denom
        decay=-group['lr']*group['weight_decay']*before
        rounding=(after-before)-updates.sum(0)-decay
        self.parts[:-2]+=updates;self.parts[-2]+=decay;self.parts[-1]+=rounding
        if not np.isfinite(self.parts).all(): raise ValueError('Nonfinite attribution')


def trace_fit(original,examples,bound,targets):
    if not examples or len(examples)!=len(bound): raise ValueError('Unaligned fitting contexts')
    torch.manual_seed(SEED)
    model=AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
    weight=model.bridge.projection.weight;weight.requires_grad_(True)
    if torch.count_nonzero(weight): raise ValueError('Expected zero bridge')
    optimizer=torch.optim.AdamW([weight],lr=1e-4,weight_decay=.01)
    origins=AdamOrigins(weight.numel(),targets)
    rng=np.random.default_rng(SEED);history=[];counts=np.zeros(targets,dtype=int)
    for epoch in range(2):
        losses,gradients=[],[]
        for j,i in enumerate(rng.permutation(len(examples))):
            e=select_arm(examples[i],bound[i],'binding');prediction=model(e)
            loss=F.smooth_l1_loss(prediction,prediction.new_tensor(e['truth']))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([weight],1.,error_if_nonfinite=True)
            before=weight.detach().double().numpy().copy()
            gradient=weight.grad.detach().double().numpy().copy()
            optimizer.step()
            origins.record(before,weight.detach().double().numpy(),gradient,
                optimizer.state[weight],optimizer.param_groups[0],e['target'])
            counts[e['target']]+=1
            losses.append(float(loss.detach()));gradients.append(float(norm))
            if (j+1)%512==0: print('E37 exact-fit',epoch+1,j+1,len(examples),flush=True)
        assert_frozen(model,original)
        history.append(dict(epoch=epoch+1,steps=len(losses),loss=float(np.mean(losses)),
            maximum_gradient=max(gradients),nonzero_gradients=sum(x>0 for x in gradients)))
    total=weight.detach().double().numpy().reshape(-1)
    error=float(abs(origins.parts.sum(0)-total).max())
    if error>1e-12: raise ValueError('Origin vectors do not sum to final weight')
    return model,origins.parts,dict(training=history,steps=origins.steps,counts=counts.tolist(),
        maximum_sum_error=error,maximum_moment_error=origins.max_moment_error,
        final_norm=float(np.linalg.norm(total)),rounding_norm=float(np.linalg.norm(origins.parts[-1])),
        decay_norm=float(np.linalg.norm(origins.parts[-2])))


def project(model,example,parts):
    with torch.inference_mode(False),torch.enable_grad():
        e=clean_example(example,torch.float64)
        zero=torch.zeros_like(model.bridge.projection.weight,requires_grad=True)
        prediction=functional_call(model,{WEIGHT:zero},(e,))
        gradient,=torch.autograd.grad(prediction,zero)
        values=parts@gradient.detach().numpy().reshape(-1)
        if not np.isfinite(values).all() or not torch.isfinite(prediction):
            raise ValueError('Nonfinite projected origin')
        return float(prediction.detach()),values


def loss_contributions(values,residual):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(residual):
        raise ValueError('Invalid directional contribution')
    total=fsum(x)
    multiplier=-1. if residual>0 else 1. if residual<0 else float(np.sign(total))
    return x*multiplier


def summarize(frame,sources):
    validate_identities(frame)
    keys=['origin_'+s for s in sources]
    if len(set(sources))!=len(sources) or set(frame.variable)-set(sources[:-2]):
        raise ValueError('Changed origin inventory')
    if frame.empty or not np.isfinite(frame[keys+['parameter_derivative']].to_numpy()).all():
        raise ValueError('Nonfinite contribution table')
    np.testing.assert_allclose(frame[keys].sum(axis=1),frame.parameter_derivative,atol=1e-9,rtol=1e-9)
    results={};maximum=0.
    for target in sources[:-2]:
        f=frame[frame.variable==target]
        if f.empty: results[target]=dict(status='unsupported');continue
        people=f.groupby('patient')[keys].mean()
        links=f.drop_duplicates('patient').set_index('patient').component
        groups=people.groupby(links.reindex(people.index)).mean()
        levels=dict(occurrence=f[keys].mean(),patient=people.mean(),component=groups.mean())
        buckets=defaultdict(list)
        for r in f.itertuples(index=False):
            buckets[r.patient].append([getattr(r,k) for k in keys])
        def mean(rows):return [fsum(x[i] for x in rows)/len(rows) for i in range(len(keys))]
        scalar_people={p:mean(v) for p,v in buckets.items()};source_groups=defaultdict(list)
        for p,v in scalar_people.items():source_groups[links[p]].append(v)
        scalar=dict(occurrence=mean([x for xs in buckets.values() for x in xs]),
            patient=mean(list(scalar_people.values())),component=mean([mean(xs) for xs in source_groups.values()]))
        entries={}
        for level,vector in levels.items():
            error=float(abs(vector.to_numpy()-scalar[level]).max());maximum=max(maximum,error)
            if error>1e-12*(1+abs(vector).max()):raise ValueError('Scalar origin aggregation mismatch')
            origins=dict(zip(sources,map(float,vector)))
            own=origins[target];other=fsum(v for s,v in origins.items() if s not in (target,*sources[-2:]))
            entries[level]=dict(origins=origins,own=own,other=other,decay=origins[sources[-2]],
                rounding=origins[sources[-1]],total=fsum(origins.values()))
        results[target]=dict(status='scored',occurrences=len(f),patients=len(people),
            components=len(groups),metrics=entries)
    return results,maximum


def classify(results,level):
    cases=dict(harmful_total=[],own_helpful_other_harmful=[],own_harmful=[],own_ambiguous=[])
    for target,r in results.items():
        if r['status']!='scored':continue
        m=r['metrics'][level];tol=1e-10*(1+fsum(abs(v) for v in m['origins'].values()))
        if m['total']<=tol:continue
        cases['harmful_total'].append(target)
        if m['own'] < -tol and m['other']>tol: cases['own_helpful_other_harmful'].append(target)
        elif m['own']>tol:cases['own_harmful'].append(target)
        else:cases['own_ambiguous'].append(target)
    return dict(counts={k:len(v) for k,v in cases.items()},targets=cases)


def run_fold(fold,parent,path_report,common):
    for stage,name in (('content_training_20261003',''),('organ_binding_20261004','_binding')):
        for role in ('fit','selection'):
            for extension in ('joblib','json'):
                p=ROOT/stage/f'fold{fold}'/f'{role}{name}.{extension}'
                if not p.is_file():raise ValueError(f'Missing original cache: {p}')
    variables,examples,bound,original,contract,extraction,_=prepare(fold)
    mp=Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
    if sha256(mp)!=parent['bindings'][str(mp)]:raise ValueError('Changed E34 fold')
    meta=json.loads(mp.read_text());check_bindings(meta['contract']['bindings']);check_bindings(meta['outputs'])
    if meta['contract']['parent']!=contract or meta['contract']['extraction']!=extraction:
        raise ValueError('Changed fitting/extraction contract')
    checkpoint=PARENT_DIR/f'fold{fold}'/'immediate.pt'
    if sha256(checkpoint)!=meta['fits']['immediate']['sha256']:raise ValueError('Changed E34 weights')
    saved=torch.load(checkpoint,weights_only=False)
    bindings=dict(common,**{str(p):sha256(p) for p in (mp,checkpoint)})
    sources=variables+['weight_decay','rounding']
    dest=DEST/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    ap,am=dest/'parts.npz',dest/'parts.json'
    if am.exists():
        info=json.loads(am.read_text())
        if info['bindings']!=bindings or info['sha256']!=sha256(ap) or info['sources']!=sources:
            raise ValueError('Changed saved origins')
        parts=np.load(ap)['parts']
    else:
        if ap.exists():raise ValueError('Unverified partial trace')
        model,parts,diag=trace_fit(original,examples['fit'],bound['fit'],len(variables))
        for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        if diag['training']!=meta['fits']['immediate']['training']:raise ValueError('Training logs differ')
        space_required();temp=ap.with_suffix('.tmp')
        with temp.open('wb') as handle:np.savez_compressed(handle,parts=parts)
        temp.replace(ap);info=dict(bindings=bindings,sha256=sha256(ap),sources=sources,diagnostic=diag,
                                  all_saved_tensors_exact=True)
        write_json(am,info)
    model=AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
    model.load_state_dict(saved,strict=True);double=deepcopy(model).double().eval().requires_grad_(False)
    np.testing.assert_allclose(parts.sum(0),saved[WEIGHT].double().numpy().reshape(-1),atol=1e-12,rtol=0)
    frames={};reports=[am];role_patients={}
    for role in ROLES:
        src=PATH_DIR/f'fold{fold}'/f'{role}.parquet'
        if sha256(src)!=path_report['bindings'][str(src)]:raise ValueError('Changed E36 queries')
        f=pd.read_parquet(src);validate_identities(f);role_patients[role]=set(f.patient)
        rb=dict(bindings,**{str(p):sha256(p) for p in (src,ap,am)})
        p,rp=dest/f'{role}.parquet',dest/f'{role}.json'
        if rp.exists():
            ri=json.loads(rp.read_text())
            if ri['bindings']!=rb or ri['sha256']!=sha256(p):raise ValueError('Changed completed origin role')
            frames[role]=pd.read_parquet(p);reports.append(rp);continue
        if p.exists():raise ValueError('Unverified partial role')
        if role=='same_source':rows=same_source_examples(f,original,variables,fold)
        else:
            expected=[(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
            if list(f[['patient','component','context','truth']].itertuples(index=False,name=None))!=expected:
                raise ValueError('Different fitting query identities')
            rows=(select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
        values=[];max_slope=max_zero=0.
        for i,e in enumerate(rows):
            zero,contributions=project(double,e,parts);old=f.iloc[i]
            err=abs(zero-old.base64);max_zero=max(max_zero,err)
            if err>1e-12:raise ValueError('Zero endpoint differs')
            err=abs(fsum(contributions)-old.slope64);max_slope=max(max_slope,err)
            if err>1e-9*(1+abs(old.slope64)):raise ValueError('Sum of origins differs from E36 tangent')
            signed=loss_contributions(contributions,old.truth-zero)
            values.append(signed)
            if (i+1)%512==0:print('E37 origins',fold,role,i+1,len(f),flush=True)
        if len(values)!=len(f):raise ValueError('Incomplete origin replay')
        for i,s in enumerate(sources):f['origin_'+s]=np.asarray(values)[:,i]
        residual=f.truth-f.base64
        f['parameter_derivative']=np.where(residual==0,abs(f.slope64),-np.sign(residual)*f.slope64)
        summarize(f,sources);space_required();tmp=p.with_suffix('.tmp');f.to_parquet(tmp,index=False);tmp.replace(p)
        write_json(rp,dict(bindings=rb,sha256=sha256(p),rows=len(f),max_slope_error=max_slope,max_zero_error=max_zero))
        frames[role]=f;reports.append(rp)
    if any(role_patients[a]&role_patients[b] for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
        raise ValueError('Role overlap')
    for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
    check_bindings(bindings)
    return frames,reports,sources,info['diagnostic']


def main():
    if OUTPUT.exists():raise ValueError('Completed diagnostic exists')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    parent=json.loads(PARENT.read_text());require_verification(PARENT,json.loads(VERIFICATION.read_text()))
    prior=json.loads(PATH_REPORT.read_text());check_bindings(prior['bindings'])
    if not prior['complete']:raise ValueError('Incomplete E36')
    paths=[PROTOCOL,Path(__file__),Path('tests/test_adam_origins.py'),PARENT,VERIFICATION,PATH_REPORT,
        Path('scripts/diagnose_parameter_path.py'),Path('scripts/train_frozen_bridge.py'),
        Path('scripts/train_organ_binding.py'),Path('scripts/train_organ_assimilation.py'),
        Path('scripts/organ_assimilation.py'),Path('whole_body_jepa/physiology_state.py')]
    common={str(p):sha256(p) for p in paths};bindings=dict(common)
    data={r:[] for r in ROLES};traces=[];sources=None
    for fold in range(5):
        frames,reports,current,trace=run_fold(fold,parent,prior,common)
        if sources is not None and sources!=current:raise ValueError('Origin order changed')
        sources=current;traces.append(trace)
        for r,f in frames.items():data[r].append(f)
        for p in reports:
            bindings[str(p)]=sha256(p)
            private=p.with_suffix('.npz' if p.stem=='parts' else '.parquet')
            bindings[str(private)]=sha256(private)
    results={};summary={};maximum=0.
    for role,parts in data.items():
        results[role],err=summarize(pd.concat(parts,ignore_index=True),sources);maximum=max(maximum,err)
        summary[role]={level:classify(results[role],level) for level in ('occurrence','patient','component')}
        print('E37',role,summary[role]['patient']['counts'],flush=True)
    check_bindings(bindings)
    write_json(OUTPUT,dict(complete=True,sources=sources,bindings=bindings,traces=traces,results=results,
        summary=summary,max_scalar_roundoff=maximum,promotion_allowed=False,runtime_modified=False,
        limitation='Local total-ray decomposition conditional on joint Adam history; not task-deletion causality, '
                   'independent model training, future forecast or untouched validation.'))


if __name__=='__main__':main()

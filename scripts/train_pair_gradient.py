"""E38: matched paired-gradient intervention in one frozen-backbone bridge."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.train_frozen_bridge import (ROOT, DEST as BASE_DIR, OUTPUT as BASE_REPORT,
    ROLES, SEED, AssimilatedReconstruction, assert_frozen, prepare, select_arm,
    space_required, write_json, check_bindings, same_source_examples, direct_prediction,
    validate_identities, paired_summary, require_verification)
from whole_body_jepa.multisource_intervals import sha256

DEST=ROOT/'pair_gradient_20261005'
PROTOCOL=Path('RECONSTRUCTION_PAIR_GRADIENT_PROTOCOL_20261005.md')
OUTPUT=Path('artifacts/native_pair_gradient_20261005.json')
CONTROLS=('mean','fixed_immediate','immediate','fixed_delayed','fixed_body','bound_start','unbridged','frozen','raw')
ARMS=('mean','pcgrad')


def combine(g,h,project):
    if g.shape!=h.shape or not g.numel() or not torch.isfinite(g).all() or not torch.isfinite(h).all():
        raise ValueError('Invalid paired gradients')
    gd,hd=g.double(),h.double()
    dot=torch.sum(gd*hd)
    if project and dot<0:
        return (((gd-dot/torch.sum(hd*hd)*hd)+(hd-dot/torch.sum(gd*gd)*gd))*.5).to(g.dtype)
    return (g+h)*.5


def fit_pair(original,examples,bound,arm):
    if arm not in ARMS or not examples or len(examples)!=len(bound) or len(examples)%2:
        raise ValueError('Aligned even nonempty fit required')
    torch.manual_seed(SEED)
    model=AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
    weight=model.bridge.projection.weight;weight.requires_grad_(True)
    opt=torch.optim.AdamW([weight],lr=1e-4,weight_decay=.01)
    rng=np.random.default_rng(SEED);log=[]
    for epoch in range(2):
        order=rng.permutation(len(examples)).reshape(-1,2);losses=[]
        stats=dict(steps=0,conflicting_pairs=0,projected_pairs=0,
                   proposed_harmful_dots=0,actual_harmful_dots=0,
                   maximum_actual_harm=0.,maximum_proposed_harm=0.)
        for step,pair in enumerate(order):
            grads=[];targets=[]
            for i in pair:
                e=select_arm(examples[i],bound[i],'binding');pred=model(e)
                loss=F.smooth_l1_loss(pred,pred.new_tensor(e['truth']))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                g,=torch.autograd.grad(loss,weight);grads.append(g.detach());targets.append(e['target'])
                losses.append(float(loss.detach()))
            conflict=float((grads[0].double()*grads[1].double()).sum())<0
            surgery=arm=='pcgrad' and targets[0]!=targets[1]
            combined=combine(*grads,surgery)
            opt.zero_grad(set_to_none=True);weight.grad=combined
            torch.nn.utils.clip_grad_norm_([weight],1.,error_if_nonfinite=True)
            proposal=weight.grad.detach().double().clone();before=weight.detach().double().clone()
            opt.step();delta=weight.detach().double()-before
            if not torch.isfinite(weight).all():raise ValueError('Nonfinite updated weights')
            for g in grads:
                g=g.double();dp=-float((g*proposal).sum());da=float((g*delta).sum())
                stats['proposed_harmful_dots']+=int(dp>1e-12*(1+float(g.norm()*proposal.norm())))
                stats['actual_harmful_dots']+=int(da>1e-12*(1+float(g.norm()*delta.norm())))
                stats['maximum_proposed_harm']=max(stats['maximum_proposed_harm'],dp)
                stats['maximum_actual_harm']=max(stats['maximum_actual_harm'],da)
            stats['steps']+=1;stats['conflicting_pairs']+=int(conflict)
            stats['projected_pairs']+=int(conflict and surgery)
            if (step+1)%512==0:print('E38 fit',arm,epoch+1,step+1,len(order),flush=True)
        assert_frozen(model,original)
        log.append(dict(epoch=epoch+1,examples=len(losses),loss=float(np.mean(losses)),**stats))
    return model,log


def describe(frame,variables):
    validate_identities(frame)
    names=[x for x in ('pcgrad',*CONTROLS) if x in frame]
    if (not set(ARMS)<=set(frame) or not np.isfinite(frame[['truth',*names]].to_numpy()).all()
            or not set(frame.variable)<=set(variables)):
        raise ValueError('Missing/nonfinite predictions')
    result={}
    for v in variables:
        f=frame[frame.variable==v]
        if f.empty:result[v]=dict(status='unsupported');continue
        people=f[names].subtract(f.truth,axis=0).abs().groupby(f.patient).mean()
        links=f.drop_duplicates('patient').set_index('patient').component
        groups=people.groupby(links.reindex(people.index)).mean()
        result[v]=dict(status='scored',patients=len(people),components=len(groups),occurrences=len(f),
            normalized_mae=dict(patient=people.mean().to_dict(),component=groups.mean().to_dict()))
    return result


def assess(f,seed):
    if f.variable.nunique()!=1 or not set(CONTROLS)<=set(f):raise ValueError('Invalid assessment')
    r=describe(f,[f.variable.iloc[0]])[f.variable.iloc[0]]
    people=f[['pcgrad',*CONTROLS]].subtract(f.truth,axis=0).abs().groupby(f.patient).mean()
    links=f.drop_duplicates('patient').set_index('patient').component
    groups=people.groupby(links.reindex(people.index)).mean();r['comparisons']={}
    for i,c in enumerate(CONTROLS):
        stats={level:paired_summary(table.pcgrad-table[c],seed+i+(100 if level=='component' else 0),5000)
               for level,table in (('patient',people),('component',groups))}
        passed=len(people)>=30 and len(groups)>=3 and all(x['ci95'] is not None and x['ci95'][1]<0
            and x['p_less'] is not None and x['p_less']<.05/(53*len(CONTROLS)) for x in stats.values())
        r['comparisons'][c]=dict(paired=stats,screen_pass=bool(passed))
    return r


def run_fold(fold,parent,common):
    for stage,suffix in (('content_training_20261003',''),('organ_binding_20261004','_binding')):
        for role in ('fit','selection'):
            for ext in ('joblib','json'):
                if not (ROOT/stage/f'fold{fold}'/f'{role}{suffix}.{ext}').is_file():raise ValueError('Missing original cache')
    variables,examples,bound,original,contract,extraction,_=prepare(fold)
    source=Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
    if sha256(source)!=parent['bindings'][str(source)]:raise ValueError('Changed E34 fold')
    old=json.loads(source.read_text());check_bindings(old['contract']['bindings']);check_bindings(old['outputs'])
    if old['contract']['parent']!=contract or old['contract']['extraction']!=extraction:raise ValueError('Changed fit contract')
    binding=dict(common,**{str(source):sha256(source)})
    dest=DEST/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'contract.json';current=dict(bindings=binding,parent=contract,extraction=extraction)
    if cp.exists() and json.loads(cp.read_text())!=current:raise ValueError('Changed experiment')
    if not cp.exists():write_json(cp,current)
    models={};reports=[cp]
    for arm in ARMS:
        p,mp=dest/f'{arm}.pt',dest/f'{arm}.json'
        if not mp.exists():
            if p.exists():raise ValueError('Unverified checkpoint')
            space_required();m,history=fit_pair(original,examples['fit'],bound['fit'],arm)
            temp=p.with_suffix('.tmp');torch.save(m.state_dict(),temp);temp.replace(p)
            write_json(mp,dict(contract=sha256(cp),sha256=sha256(p),arm=arm,training=history))
        meta=json.loads(mp.read_text())
        if meta['contract']!=sha256(cp) or meta['sha256']!=sha256(p) or meta['arm']!=arm:raise ValueError('Changed checkpoint')
        m=AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
        m.load_state_dict(torch.load(p,weights_only=False),strict=True);assert_frozen(m,original)
        models[arm]=m;reports.extend((p,mp))
    frames={};patients={}
    for role in ROLES:
        src=BASE_DIR/f'fold{fold}'/f'{role}.parquet'
        if sha256(src)!=old['outputs'][str(src)]:raise ValueError('Changed original rows')
        f=pd.read_parquet(src);validate_identities(f);patients[role]=set(f.patient)
        if len(f)!=(2048 if role=='fit' else 1024):raise ValueError('Changed role size')
        p,mp=dest/f'{role}.parquet',dest/f'{role}.json'
        rb={str(x):sha256(x) for x in (src,*reports)}
        if mp.exists():
            meta=json.loads(mp.read_text())
            if meta['bindings']!=rb or meta['sha256']!=sha256(p):raise ValueError('Changed completed role')
            frames[role]=pd.read_parquet(p);continue
        if p.exists():raise ValueError('Unverified role')
        if role=='same_source':rows=same_source_examples(f,original,variables,fold)
        else:
            expected=[(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
            if list(f[['patient','component','context','truth']].itertuples(index=False,name=None))!=expected:raise ValueError('Changed identity')
            rows=(select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
        points=[];parity=0.
        with torch.inference_mode():
            for i,e in enumerate(rows):
                out={}
                for arm,m in models.items():
                    value=float(m(e));check=direct_prediction(m,e,e['organs'])
                    np.testing.assert_allclose(value,check,atol=1e-7,rtol=0)
                    parity=max(parity,abs(value-check));out[arm]=value
                points.append(out)
                if (i+1)%512==0:print('E38 replay',fold,role,i+1,len(f),flush=True)
        if len(points)!=len(f):raise ValueError('Incomplete query replay')
        for arm in ARMS:f[arm]=[p[arm] for p in points]
        describe(f,variables);space_required();temp=p.with_suffix('.tmp');f.to_parquet(temp,index=False);temp.replace(p)
        write_json(mp,dict(bindings=rb,sha256=sha256(p),parity=parity,rows=len(f)))
        frames[role]=f
    if any(patients[a]&patients[b] for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):raise ValueError('Role overlap')
    for arm,m in models.items():
        saved=torch.load(dest/f'{arm}.pt',weights_only=False)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
    check_bindings(binding)
    reports.extend(dest/f'{r}.{ext}' for r in ROLES for ext in ('parquet','json'))
    return frames,variables,reports


def main():
    if OUTPUT.exists():raise ValueError('Completed experiment exists')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    verification=Path('artifacts/native_frozen_bridge_verification_20261005.json')
    parent=json.loads(BASE_REPORT.read_text());require_verification(BASE_REPORT,json.loads(verification.read_text()))
    paths=[PROTOCOL,Path(__file__),Path('tests/test_pair_gradient.py'),BASE_REPORT,verification,
        Path('scripts/train_frozen_bridge.py'),Path('scripts/train_organ_binding.py'),
        Path('scripts/verify_organ_assimilation.py'),Path('scripts/audit_trained_bridge.py'),
        Path('scripts/organ_assimilation.py'),Path('scripts/train_organ_assimilation.py'),
        Path('whole_body_jepa/physiology_state.py'),
        Path('scripts/score_additional_reconstruction.py'),Path('artifacts/native_adam_origins_20261005.json')]
    common={str(p):sha256(p) for p in paths};bindings=dict(common);data={r:[] for r in ROLES};inventory=None
    for fold in range(5):
        frames,variables,reports=run_fold(fold,parent,common)
        if inventory is not None and inventory!=variables:raise ValueError('Changed inventory')
        inventory=variables
        for r in ROLES:data[r].append(frames[r])
        bindings.update({str(p):sha256(p) for p in reports})
    frames={r:pd.concat(fs,ignore_index=True) for r,fs in data.items()}
    results={r:describe(f,inventory) for r,f in frames.items()}
    for i,v in enumerate(inventory):
        f=frames['selection'][frames['selection'].variable==v]
        if len(f):results['selection'][v]=assess(f,SEED+i*1000)
    screens={c:[v for v,r in results['selection'].items() if r['status']=='scored' and r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    check_bindings(bindings)
    summary=dict(screens=screens,full_screen=sorted(set.intersection(*(set(x) for x in screens.values()))),occurrences={r:len(f) for r,f in frames.items()})
    write_json(OUTPUT,dict(complete=True,results=results,summary=summary,bindings=bindings,
        independent_verification=False,runtime_modified=False,promotion_allowed=False,
        limitation='Fixed paired-gradient intervention; reused development, current reconstruction, no deployment claim.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

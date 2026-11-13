"""Independent E40 replay, scalar scoring and finite-loss verification."""
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts import train_actual_step as audit
from scripts.verify_frozen_bridge import mean, paired, compare

OUTPUT = Path('artifacts/native_actual_step_verification_20261005.json')


def scalar_results(frame, inventory, selection=False):
    audit.base.validate_identities(frame)
    names = [n for n in (audit.PRIMARY, *audit.CONTROLS) if n in frame]
    if not {audit.PRIMARY,'pcgrad','mean'} <= set(names) or (selection and not set(audit.CONTROLS) <= set(names)):
        raise ValueError('Missing predictions')
    targets = defaultdict(list)
    for row in frame.to_dict('records'):
        if row['variable'] not in inventory or not all(isfinite(row[n]) for n in ('truth',*names)):
            raise ValueError('Invalid values/inventory')
        targets[row['variable']].append(row)
    result = {}
    for i, variable in enumerate(inventory):
        rows = targets[variable]
        if not rows:
            result[variable] = dict(status='unsupported'); continue
        people, links = defaultdict(list), {}
        for r in rows:
            people[r['patient']].append(r); links[r['patient']] = r['component']
        losses = {p:{n:mean([abs(r[n]-r['truth']) for r in people[p]]) for n in names} for p in sorted(people)}
        members = defaultdict(list)
        for p, loss in losses.items(): members[links[p]].append(loss)
        groups = {c:{n:mean([r[n] for r in members[c]]) for n in names} for c in sorted(members)}
        entry = dict(status='scored',patients=len(losses),components=len(groups),occurrences=len(rows),
            normalized_mae={s:{n:mean([r[n] for r in table.values()]) for n in names}
                            for s,table in (('patient',losses),('component',groups))})
        if selection:
            entry['comparisons'] = {}
            for j, control in enumerate(audit.CONTROLS):
                stats = {s:paired([r[audit.PRIMARY]-r[control] for r in table.values()],
                                 audit.base.SEED+i*1000+j+offset,5000)
                         for s,table,offset in (('patient',losses,0),('component',groups,100))}
                passed = len(losses)>=30 and len(groups)>=3 and all(
                    r['ci95'] is not None and r['ci95'][1]<0 and r['p_less'] is not None
                    and r['p_less']<.05/(53*len(audit.CONTROLS)) for r in stats.values())
                entry['comparisons'][control] = dict(paired=stats,screen_pass=bool(passed))
        result[variable] = entry
    return result


def verify_trace(frame):
    if frame.empty or frame.duplicated(['fold','epoch','step','slot']).any():
        raise ValueError('Invalid trace identity')
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all(): raise ValueError('Nonfinite trace')
    for name in ('raw_tolerance','actual_tolerance'):
        if (frame[name]<=0).any(): raise ValueError('Invalid tolerance')
    for name in ('raw_norm','actual_norm','correction_norm'):
        if (frame[name]<0).any(): raise ValueError('Negative norm')
    for suffix in ('before','after'):
        pred = torch.tensor(frame['prediction_'+suffix].to_numpy(),dtype=torch.float32)
        truth = torch.tensor(frame.truth.to_numpy(),dtype=torch.float32)
        np.testing.assert_array_equal(F.smooth_l1_loss(pred,truth,reduction='none').double().numpy(),
                                      frame['loss_'+suffix].to_numpy())
    if (frame.cross_target & (frame.actual_dot>frame.actual_tolerance)).any():
        raise ValueError('Protected local constraint violated')
    if ((~frame.cross_target) & frame.projected).any() or (frame.rounding_veto & ~frame.projected).any():
        raise ValueError('Invalid projection decision')
    if ((~frame.projected) & (frame.correction_norm!=0)).any() or (frame.rounding_veto & (frame.actual_norm!=0)).any():
        raise ValueError('Invalid unmodified/vetoed step')
    result = {}
    rows = frame.to_dict('records')
    for name, records in (('all',rows),('cross_target',[r for r in rows if r['cross_target']]),
                           ('same_target',[r for r in rows if not r['cross_target']])):
        steps = {}
        for r in records:
            key = (r['fold'],r['epoch'],r['step'])
            if key in steps:
                for field in ('projected','rounding_veto','raw_norm','actual_norm','correction_norm'):
                    if steps[key][field] != r[field]: raise ValueError('Slot step mismatch')
            steps[key] = r
        change = [r['loss_after']-r['loss_before'] for r in records]
        tol = [1e-12*(1+abs(r['loss_before'])+abs(r['loss_after'])) for r in records]
        result[name] = dict(examples=len(records),steps=len(steps),
            projected_steps=sum(r['projected'] for r in steps.values()),
            rounding_veto_steps=sum(r['rounding_veto'] for r in steps.values()),
            raw_harm=sum(r['raw_dot']>r['raw_tolerance'] for r in records),
            actual_harm=sum(r['actual_dot']>r['actual_tolerance'] for r in records),
            finite_harm=sum(d>t for d,t in zip(change,tol)),finite_help=sum(d<-t for d,t in zip(change,tol)),
            finite_tie=sum(abs(d)<=t for d,t in zip(change,tol)),
            mean_loss_change=mean(change) if records else None,
            mean_mae_change=mean([abs(r['prediction_after']-r['truth'])-abs(r['prediction_before']-r['truth']) for r in records]) if records else None,
            mean_raw_norm=mean([r['raw_norm'] for r in steps.values()]) if steps else None,
            mean_actual_norm=mean([r['actual_norm'] for r in steps.values()]) if steps else None)
    return result


def diagnosis(results):
    result = {}
    for role, all_targets in results.items():
        targets = {v:r for v,r in all_targets.items() if r['status']=='scored'}
        names = next(iter(targets.values()))['normalized_mae']['patient']
        result[role] = dict(targets=len(targets),macro={scope:{n:mean([r['normalized_mae'][scope][n]
            for r in targets.values()]) for n in names} for scope in ('patient','component')},contrasts={})
        for control in names:
            if control == audit.PRIMARY: continue
            deltas = {}
            for v,r in targets.items():
                f = results['fit'][v]
                fit = None if f['status']!='scored' or control not in f['normalized_mae']['patient'] else (
                    f['normalized_mae']['patient'][audit.PRIMARY]-f['normalized_mae']['patient'][control])
                deltas[v] = dict(query=r['normalized_mae']['patient'][audit.PRIMARY]-r['normalized_mae']['patient'][control],fit=fit)
            result[role]['contrasts'][control] = dict(mean_gain=sum(r['query']<0 for r in deltas.values()),
                fit_gain_query_non_gain=sum(r['fit'] is not None and r['fit']<0 and r['query']>=0 for r in deltas.values()),targets=deltas)
    return result


def main():
    if OUTPUT.exists(): raise ValueError('Completed verification exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    report = json.loads(audit.OUTPUT.read_text())
    if not report['complete'] or report['runtime_modified'] or report['promotion_allowed']:
        raise ValueError('Invalid completion flags')
    audit.base.check_bindings(report['bindings'])
    old = json.loads(audit.base.OUTPUT.read_text())
    audit.base.check_bindings(old['bindings'])
    paths = [audit.OUTPUT,Path(__file__),Path('tests/test_actual_step_verification.py'),Path('scripts/verify_frozen_bridge.py')]
    data = {r:[] for r in audit.base.ROLES}; traces=[]; inventory=None; parity={}; error=0.
    for fold in range(5):
        variables,examples,bound,original,parent,extraction,_ = audit.prepare_existing(fold)
        if inventory is not None and inventory != variables: raise ValueError('Inventory changed')
        inventory=variables;dest=audit.DEST/f'fold{fold}'
        cp,model_path,meta_path,trace_path=[dest/n for n in ('contract.json','actual_step.pt','actual_step.json','training.parquet')]
        contract=json.loads(cp.read_text());meta=json.loads(meta_path.read_text())
        if contract['parent']!=parent or contract['extraction']!=extraction: raise ValueError('Changed extraction')
        audit.base.check_bindings(contract['bindings'])
        if meta['contract']!=audit.base.sha256(cp) or meta['sha256']!=audit.base.sha256(model_path) or meta['trace_sha256']!=audit.base.sha256(trace_path):
            raise ValueError('Changed model/trace')
        trace=pd.read_parquet(trace_path)
        expected={(fold,e,s,i) for e in (1,2) for s in range(1024) for i in (0,1)}
        if len(trace)!=4096 or set(trace[['fold','epoch','step','slot']].itertuples(index=False,name=None))!=expected:
            raise ValueError('Incomplete training trace')
        if not set(trace.target)<=set(range(len(inventory))): raise ValueError('Bad trace target')
        error=max(error,compare(meta['trace_summary'],verify_trace(trace)))
        rng=np.random.default_rng(audit.base.SEED)
        for epoch,log in zip((1,2),meta['training'],strict=True):
            part=trace[trace.epoch==epoch].sort_values(['step','slot'])
            indices=rng.permutation(2048)
            if list(part.target)!=[examples['fit'][i]['target'] for i in indices]: raise ValueError('Changed pair order')
            np.testing.assert_array_equal(part.truth,[examples['fit'][i]['truth'] for i in indices])
            expected_cross=np.repeat([examples['fit'][a]['target']!=examples['fit'][b]['target'] for a,b in indices.reshape(-1,2)],2)
            np.testing.assert_array_equal(part.cross_target,expected_cross)
            error=max(error,compare(log,dict(epoch=epoch,steps=1024,examples=2048,loss=mean(part.loss_before.tolist()))))
        traces.append(trace)
        model=audit.base.AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
        saved=torch.load(model_path,weights_only=False);model.load_state_dict(saved,strict=True)
        audit.base.assert_frozen(model,original)
        if not all(torch.isfinite(v).all() for v in saved.values()): raise ValueError('Nonfinite weights')
        people={};parity[str(fold)]={}
        for role in audit.base.ROLES:
            p,mp=dest/f'{role}.parquet',dest/f'{role}.json'
            src=audit.base.DEST/f'fold{fold}'/f'{role}.parquet'
            f,prior=pd.read_parquet(p),pd.read_parquet(src)
            m=json.loads(mp.read_text())
            if m['bindings']!={str(x):audit.base.sha256(x) for x in (src,cp,model_path,meta_path)} or m['sha256']!=audit.base.sha256(p):
                raise ValueError('Changed role bindings')
            pd.testing.assert_frame_equal(f[list(prior)],prior,check_exact=True)
            if len(f)!=(2048 if role=='fit' else 1024) or m['rows']!=len(f): raise ValueError('Changed support')
            audit.base.validate_identities(f);people[role]=set(f.patient)
            if role=='same_source': rows=audit.base.same_source_examples(f,original,variables,fold)
            else:
                if list(f[['patient','component','context','truth']].itertuples(index=False,name=None)) != [
                        (e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]:
                    raise ValueError('Changed identity')
                rows=(audit.base.select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
            maximum=0.;count=0
            with torch.inference_mode():
                for i,e in enumerate(rows):
                    value=audit.base.direct_prediction(model,e,e['organs'])
                    if not isfinite(value): raise ValueError('Nonfinite replay')
                    delta=abs(value-f[audit.PRIMARY].iloc[i]);maximum=max(maximum,delta);count+=1
                    if delta>1e-7: raise ValueError('Replay mismatch')
            if count!=len(f): raise ValueError('Incomplete replay')
            parity[str(fold)][role]=maximum;data[role].append(f);paths.extend((p,mp,src))
            print('E40 verified replay',fold,role,count,maximum,flush=True)
        if any(people[a]&people[b] for a,b in (('fit','selection'),('fit','same_source'),('selection','same_source'))):
            raise ValueError('Role overlap')
        for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        paths.extend((cp,model_path,meta_path,trace_path))
    frames={r:pd.concat(fs,ignore_index=True) for r,fs in data.items()}
    results={r:scalar_results(f,inventory,r=='selection') for r,f in frames.items()}
    error=max(error,compare(report['results'],results))
    training=verify_trace(pd.concat(traces,ignore_index=True));error=max(error,compare(report['training'],training))
    screens={c:[v for v,r in results['selection'].items() if r['status']=='scored' and r['comparisons'][c]['screen_pass']] for c in audit.CONTROLS}
    summary=dict(screens=screens,full_screen=sorted(set.intersection(*(set(x) for x in screens.values()))),
                 occurrences={r:len(f) for r,f in frames.items()})
    compare(report['summary'],summary);audit.base.check_bindings(report['bindings'])
    audit.base.write_json(OUTPUT,dict(complete=True,bindings={str(p):audit.base.sha256(p) for p in paths},
        maximum_aggregation_roundoff=error,parity=parity,training=training,summary=summary,
        diagnosis=diagnosis(results),runtime_modified=False,promotion_allowed=False))
    print('E40 independently verified',error,summary,flush=True)


if __name__=='__main__':main()

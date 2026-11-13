"""Independent scalar E43 scoring and saved-checkpoint replay."""
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import fit_tangent_bridge as audit
from scripts.verify_frozen_bridge import mean, paired, compare

OUTPUT = Path('artifacts/native_tangent_bridge_verification_20261005.json')


def scalar_results(frame, inventory, selection=False):
    audit.base.validate_identities(frame)
    names = [n for n in (audit.PRIMARY,'local_linear',*audit.CONTROLS) if n in frame]
    if not {audit.PRIMARY,'local_linear','bound_start'} <= set(names) or (selection and not set(audit.CONTROLS) <= set(names)):
        raise ValueError('Missing model columns')
    targets = defaultdict(list)
    for row in frame.to_dict('records'):
        if row['variable'] not in inventory or not all(isfinite(row[k]) for k in ('truth',*names)):
            raise ValueError('Invalid values/inventory')
        targets[row['variable']].append(row)
    output = {}
    for i,v in enumerate(inventory):
        rows = targets[v]
        if not rows:
            output[v] = dict(status='unsupported'); continue
        people,links = defaultdict(list),{}
        for r in rows: people[r['patient']].append(r); links[r['patient']] = r['component']
        result = dict(status='scored',patients=len(people),components=len(set(links.values())),occurrences=len(rows))
        tables = {}
        for metric in ('mae','smooth_l1'):
            def loss(row,name):
                e = abs(row[name]-row['truth'])
                return e if metric == 'mae' else (.5*e*e if e<1 else e-.5)
            losses = {p:{n:mean([loss(r,n) for r in people[p]]) for n in names} for p in sorted(people)}
            members = defaultdict(list)
            for p,r in losses.items(): members[links[p]].append(r)
            groups = {c:{n:mean([r[n] for r in members[c]]) for n in names} for c in sorted(members)}
            result[metric] = {level:{n:mean([r[n] for r in values.values()]) for n in names}
                for level,values in (('patient',losses),('component',groups))}
            if metric == 'mae': tables = dict(patient=losses,component=groups)
        if selection:
            result['comparisons'] = {}
            for j,c in enumerate(audit.CONTROLS):
                stats = {level:paired([r[audit.PRIMARY]-r[c] for r in tables[level].values()],
                    audit.base.SEED+i*1000+j+offset,5000) for level,offset in (('patient',0),('component',100))}
                passed = len(people)>=30 and len(set(links.values()))>=3 and all(
                    s['ci95'] is not None and s['ci95'][1]<0 and s['p_less'] is not None
                    and s['p_less']<.05/(53*len(audit.CONTROLS)) for s in stats.values())
                result['comparisons'][c] = dict(paired=stats,screen_pass=bool(passed))
        output[v] = result
    return output


def check_line(info, frame, direction):
    a = info['accepted_alpha']; solver = info['solver']; n = len(frame)
    if not solver['converged'] or solver['gradient_inf'] > 1e-5:
        raise ValueError('Unaccepted solve')
    def smooth(residual):
        x=abs(float(residual)); return .5*x*x if x<1 else x-.5
    origin=mean([smooth(r.bound_start-r.truth) for r in frame.itertuples()])
    linear=mean([smooth(r.tangent_base+r.tangent_slope-r.truth) for r in frame.itertuples()])
    penalty=solver['regularization']*float(direction@direction)/(2*n)
    np.testing.assert_allclose(origin,info['line_search']['origin'],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(linear,info['linear_fit_loss'],atol=1e-12,rtol=1e-12)
    # The solver report is float64; the saved, evaluated direction is float32.
    np.testing.assert_allclose(linear+penalty,solver['final_objective'],atol=1e-5,rtol=1e-5)
    derivative=mean([max(-1.,min(1.,r.bound_start-r.truth))*r.tangent_slope for r in frame.itertuples()])
    np.testing.assert_allclose(derivative,info['line_search']['derivative'],atol=1e-12,rtol=1e-12)
    trials=info['line_search']['trials']
    if [r['alpha'] for r in trials] != list(audit.STEPS[:len(trials)]):
        raise ValueError('Changed step order')
    if len(trials)>len(audit.STEPS) or any(r['accepted'] for r in trials[:-1]):
        raise ValueError('Skipped earlier accepted step')
    for t in trials:
        accepted=t['objective']<=origin+1e-4*t['alpha']*derivative
        if accepted!=t['accepted']: raise ValueError('Wrong Armijo decision')
    if a:
        actual=mean([smooth(r.tangent_step-r.truth) for r in frame.itertuples()])+a*a*penalty
        if not trials or not trials[-1]['accepted'] or a!=trials[-1]['alpha'] or derivative>=0:
            raise ValueError('Invalid accepted step')
        np.testing.assert_allclose(actual,trials[-1]['objective'],atol=1e-12,rtol=1e-12)
    else:
        if any(t['accepted'] for t in trials) or (derivative<0 and len(trials)!=len(audit.STEPS)):
            raise ValueError('Improper zero fallback')
        np.testing.assert_array_equal(frame.tangent_step,frame.bound_start)
    np.testing.assert_allclose(frame.local_linear,frame.tangent_base+a*frame.tangent_slope,atol=0,rtol=0)


def main():
    if OUTPUT.exists(): raise ValueError('Completed verification exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    report=json.loads(audit.OUTPUT.read_text())
    if not report['complete'] or report['runtime_modified'] or report['promotion_allowed']:
        raise ValueError('Wrong report state')
    audit.base.check_bindings(report['bindings'])
    paths=[audit.OUTPUT,Path(__file__),Path('tests/test_tangent_verification.py')]
    data={r:[] for r in audit.base.ROLES}; replays=[]
    for fold in range(5):
        variables,examples,bound,original,contract,extraction,_=audit.previous.prepare_existing(fold)
        dest=audit.DEST/f'fold{fold}'
        info=json.loads((dest/'complete.json').read_text());audit.base.check_bindings(info['outputs'])
        model=audit.base.AssimilatedReconstruction(original,'immediate').eval().requires_grad_(False)
        model.load_state_dict(torch.load(dest/'tangent_step.pt',weights_only=False),strict=True)
        audit.base.assert_frozen(model,original)
        direction=torch.load(dest/'direction.pt',weights_only=False).double().numpy()
        expected=torch.tensor(info['accepted_alpha']*direction,dtype=torch.float32).reshape_as(model.bridge.projection.weight)
        torch.testing.assert_close(model.bridge.projection.weight,expected,atol=0,rtol=0)
        for role in audit.base.ROLES:
            f=pd.read_parquet(dest/f'{role}.parquet')
            old=pd.read_parquet(audit.previous.DEST/f'fold{fold}'/f'{role}.parquet')
            pd.testing.assert_frame_equal(f[list(old)],old,check_exact=True)
            if role=='same_source': rows=audit.base.same_source_examples(f,original,variables,fold)
            else:
                identities=[(e['patient'],e['component'],e['target'],e['truth']) for e in examples[role]]
                if identities!=list(f[['patient','component','context','truth']].itertuples(index=False,name=None)):
                    raise ValueError('Changed patient queries')
                rows=(audit.base.select_arm(e,b,'binding') for e,b in zip(examples[role],bound[role]))
            maximum=0.;count=0
            with torch.inference_mode():
                for count,e in enumerate(rows,1):
                    point=audit.base.direct_prediction(model,e,e['organs'])
                    error=abs(point-f.tangent_step.iloc[count-1]);maximum=max(maximum,error)
                    if error>1e-7: raise ValueError('Saved checkpoint replay mismatch')
            if count!=len(f): raise ValueError('Incomplete replay')
            if role=='fit':check_line(info,f,direction)
            data[role].append(f); replays.append(dict(fold=fold,role=role,rows=count,max_error=maximum))
            print('E43 verified',fold,role,count,maximum,flush=True)
        paths.extend([dest/'complete.json',*map(Path,info['outputs'])])
    maximum=0.;actual={}
    for role,parts in data.items():
        f=pd.concat(parts,ignore_index=True)
        actual[role]=scalar_results(f,variables,selection=role=='selection')
        maximum=max(maximum,compare(actual[role],report['results'][role]))
    screens={c:[v for v,r in actual['selection'].items() if r['status']=='scored' and r['comparisons'][c]['screen_pass']]
        for c in audit.CONTROLS}
    compare(screens,report['summary']['screens'])
    compare(sorted(set.intersection(*(set(v) for v in screens.values()))),report['summary']['full_screen'])
    audit.base.check_bindings(report['bindings'])
    audit.base.write_json(OUTPUT,dict(complete=True,bindings={str(p):audit.base.sha256(p) for p in paths},
        replay=replays,scalar_and_statistics_max_error=maximum,runtime_modified=False,promotion_allowed=False))
    print('E43 independent verification complete',maximum,flush=True)


if __name__=='__main__':main()

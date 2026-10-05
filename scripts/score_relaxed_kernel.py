"""Reuse identical paired statistics, binding candidate and old-map roles explicitly."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.score_scaled_kernel import assess as paired_assess, FAMILY
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_relaxed_kernel import ARMS,DEST,OLD_DEST,ROOT,MULTIPLIER
from whole_body_jepa.multisource_intervals import sha256

BASE=('native','body_probe','raw_probe','population')
OLD=tuple(a.replace('relaxed_','scaled_') for a in ARMS)
ALL_ARMS=(*ARMS,*BASE,*OLD)
NAMES={**{a:a.replace('scaled_','relaxed_') for a in OLD},
       **{a.replace('scaled_','kernel_'):a for a in OLD}}


def assess(frame,seed,repetitions=5000):
    # The original pure statistic uses scaled_* for candidates and kernel_* for
    # their prior-map references. Bind those roles without changing its gate.
    columns=['patient','fold','component','target','truth',*ALL_ARMS]
    inverse={v:k for k,v in NAMES.items()}
    statistical=frame[columns].rename(columns=inverse)
    r=paired_assess(statistical,seed,repetitions)
    def names(values): return {NAMES.get(k,k):v for k,v in values.items()}
    r['normalized_mae']={l:names(v) for l,v in r['normalized_mae'].items()}
    r['candidates']={NAMES[a]:dict(v,comparisons=names(v['comparisons'])) for a,v in r['candidates'].items()}
    for row in r['per_component']: row['normalized_mae']=names(row['normalized_mae'])
    return r


def main():
    out=Path('artifacts/native_reconstruction_relaxed_kernel_assessment_20261003.json')
    if out.exists(): raise ValueError('Existing assessment')
    if FAMILY!=2*47*7: raise ValueError('Changed statistical family')
    op=Path('artifacts/native_reconstruction_scaled_kernel_assessment_20261003.json')
    original=json.loads(op.read_text()); checked={}; verify_bound_files(original['bindings'],checked)
    targets=sorted(original['targets'])
    if len(targets)!=47: raise ValueError('Incomplete family')
    fp=ROOT/'partitions_20260922.pt'; roles=torch.load(fp,weights_only=False)['folds']
    paths=[Path(__file__),Path('scripts/score_scaled_kernel.py'),Path('scripts/score_additional_reconstruction.py'),op,fp]
    frames,diagnostics=[],{}
    for f in range(5):
        p=Path(f'artifacts/native_reconstruction_relaxed_kernel_fold{f}_20261003.json')
        report=json.loads(p.read_text()); verify_bound_files(report['bindings'],checked)
        cp=DEST/f'fold{f}/contract.json'
        if (report['fold']!=f or report['C_multiplier']!=MULTIPLIER or sorted(report['targets'])!=targets
                or json.loads(cp.read_text())['bindings']!=report['bindings']): raise ValueError('Changed contract')
        paths.extend((p,cp)); diagnostics[str(f)]={}
        for t,meta in report['targets'].items():
            qp,mp=[DEST/f'fold{f}/{t}.{s}' for s in ('parquet','joblib')]
            if (sha256(qp)!=meta['prediction_sha256'] or sha256(mp)!=meta['model_sha256']
                    or sha256(cp)!=meta['contract_sha256']): raise ValueError('Changed artifact')
            frame=pd.read_parquet(qp); old=pd.read_parquet(OLD_DEST/f'fold{f}/{t}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient)!=set(old.patient)
                    or any(roles[f][i]!='selection' for i in frame.patient) or not (frame.fold==f).all()
                    or not (frame.target==t).all() or len(frame)!=meta['selection_patients']): raise ValueError('Changed query')
            old=old.set_index('patient').loc[frame.patient]
            for c in ('truth','component',*BASE,*OLD):
                if not np.array_equal(frame[c],old[c]): raise ValueError('Changed comparison')
            frames.append(frame); paths.extend((qp,mp)); diagnostics[str(f)][t]=meta['checks']
    data=pd.concat(frames,ignore_index=True); results={}
    for i,t in enumerate(targets):
        r=assess(data.loc[data.target==t],20261003+1000*i)
        for l in ('patient','component'):
            for a in (*BASE,*OLD):
                if not np.isclose(r['normalized_mae'][l][a],original['targets'][t]['normalized_mae'][l][a],atol=1e-12,rtol=1e-12): raise ValueError('Aggregate parity failed')
        r['all_solvers_converged']=all(diagnostics[str(f)][t][a]['converged'] for f in range(5) for a in ARMS)
        for arm in r['candidates']: r['candidates'][arm]['full_screen'] &= r['all_solvers_converged']
        results[t]=r; print('Scored relaxed kernel',i+1,'of47',flush=True)
    if len(data)!=original['summary']['query_occurrences']: raise ValueError('Incomplete grid')
    candidates=('relaxed_body','relaxed_raw')
    summary=dict(query_occurrences=len(data),unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient','target']].drop_duplicates()),
        full_screen={a:[t for t,r in results.items() if r['candidates'][a]['full_screen']] for a in candidates},
        nonconverged_fits=sum(not c['converged'] for f in diagnostics.values() for t in f.values() for c in t.values()),
        fit_mean_improvements={a:sum(t[a]['fit_mae']<t[a]['old_fit_mae'] for f in diagnostics.values() for t in f.values()) for a in ARMS},
        comparison_screens={a:{ref:[t for t,r in results.items() if r['candidates'][a]['comparisons'][ref]['screen'] and r['all_solvers_converged']]
                            for ref in next(iter(results.values()))['candidates'][a]['comparisons']} for a in candidates},
        normalized_target_macro={l:{a:float(np.mean([r['normalized_mae'][l][a] for r in results.values()])) for a in ALL_ARMS} for l in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,targets=results,diagnostics=diagnostics,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,limitation='Single fixed C-times-ten on identical maps. Reused source only. No external/forecast/interval/promotion.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

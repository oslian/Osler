"""Two preregistered scale-repair comparisons with patient/component inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_scaled_kernel import ARMS, DEST, SETTINGS, OLD_DEST, ROOT
from whole_body_jepa.multisource_intervals import sha256

CANDIDATES=('scaled_body','scaled_raw')
COMMON=('native','body_probe','raw_probe','population','scaled_process','scaled_shuffle')
ALL_ARMS=tuple(dict.fromkeys((*ARMS,*COMMON,'kernel_body','kernel_raw','kernel_process','kernel_shuffle')))
FAMILY=2*47*7


def assess(frame,seed,repetitions=5000):
    if (frame.empty or frame.target.nunique()!=1 or frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth',*ALL_ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired scaled-kernel frame')
    errors=frame[list(ALL_ARMS)].subtract(frame.truth,axis=0).abs()
    errors['patient']=frame.patient.to_numpy()
    people=errors.groupby('patient')[list(ALL_ARMS)].mean()
    people['component']=frame.drop_duplicates('patient').set_index('patient').component.reindex(people.index)
    groups=people.groupby('component')[list(ALL_ARMS)].mean()
    candidates={}
    for k,arm in enumerate(CANDIDATES):
        comparisons={}
        for i,ref in enumerate((*COMMON,arm.replace('scaled_','kernel_'))):
            paired={level:paired_summary(rows[arm]-rows[ref],seed+k*100+10*i+j,repetitions)
                    for j,(level,rows) in enumerate((('patient',people),('component',groups)))}
            passed=len(people)>=30 and len(groups)>=3 and all(s['mean']<0 and s['p_less'] is not None
                and s['p_less']<.05/FAMILY and s['ci95'] is not None and s['ci95'][1]<0 for s in paired.values())
            comparisons[ref]=dict(paired=paired,screen=bool(passed),
                                  component_mean_harms=int((groups[arm]>groups[ref]).sum()))
        candidates[arm]=dict(comparisons=comparisons,full_screen=all(c['screen'] for c in comparisons.values()))
    return dict(query_occurrences=len(frame),unique_patients=len(people),components=len(groups),
        normalized_mae=dict(patient=people[list(ALL_ARMS)].mean().to_dict(),component=groups.mean().to_dict()),
        candidates=candidates,
        per_component=[dict(component=str(c),patients=int((people.component==c).sum()),normalized_mae=row.to_dict())
                       for c,row in groups.iterrows()])


def main():
    out=Path('artifacts/native_reconstruction_scaled_kernel_assessment_20261003.json')
    if out.exists(): raise ValueError('Existing complete assessment')
    op=Path('artifacts/native_reconstruction_kernel_assessment_20261003.json')
    original=json.loads(op.read_text()); checked={}; verify_bound_files(original['bindings'],checked)
    targets=sorted(original['targets'])
    if len(targets)!=47: raise ValueError('Incomplete targets')
    fp=ROOT/'partitions_20260922.pt'; roles=torch.load(fp,weights_only=False)['folds']
    paths=[Path(__file__),Path('scripts/score_additional_reconstruction.py'),fp,op]
    frames,diagnostics=[],{}
    for f in range(5):
        p=Path(f'artifacts/native_reconstruction_scaled_kernel_fold{f}_20261003.json')
        report=json.loads(p.read_text()); verify_bound_files(report['bindings'],checked)
        cp=DEST/f'fold{f}/contract.json'
        if (report['fold']!=f or report['settings']!=SETTINGS or sorted(report['targets'])!=targets
                or json.loads(cp.read_text())['bindings']!=report['bindings']): raise ValueError('Changed fold contract')
        paths.extend((p,cp)); diagnostics[str(f)]={}
        for t,meta in report['targets'].items():
            qp,mp=[DEST/f'fold{f}/{t}.{s}' for s in ('parquet','joblib')]
            if (meta['prediction_sha256']!=sha256(qp) or meta['model_sha256']!=sha256(mp)
                    or meta['contract_sha256']!=sha256(cp)): raise ValueError('Changed artifact')
            frame=pd.read_parquet(qp); old=pd.read_parquet(OLD_DEST/f'fold{f}/{t}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient)!=set(old.patient)
                    or any(roles[f][i]!='selection' for i in frame.patient) or not (frame.fold==f).all()
                    or not (frame.target==t).all() or len(frame)!=meta['selection_patients']): raise ValueError('Changed query')
            old=old.set_index('patient').loc[frame.patient]
            for c in ('truth','component',*COMMON[:4],'kernel_body','kernel_raw','kernel_process','kernel_shuffle'):
                if not np.array_equal(frame[c],old[c]): raise ValueError('Changed reference')
            frames.append(frame); paths.extend((qp,mp)); diagnostics[str(f)][t]=meta['checks']
    data=pd.concat(frames,ignore_index=True); results={}
    for i,t in enumerate(targets):
        r=assess(data.loc[data.target==t],20261003+1000*i)
        for l in ('patient','component'):
            for a in ('native','body_probe','raw_probe','population','kernel_body','kernel_raw'):
                if not np.isclose(r['normalized_mae'][l][a],original['targets'][t]['normalized_mae'][l][a],atol=1e-12,rtol=1e-12):
                    raise ValueError('All-fold parity failed')
        r['all_solvers_converged']=all(diagnostics[str(f)][t][a]['converged'] for f in range(5) for a in ARMS)
        for arm in CANDIDATES: r['candidates'][arm]['full_screen'] &= r['all_solvers_converged']
        results[t]=r
        print('Scored scaled kernel',i+1,'of47',flush=True)
    checks=[c for f in diagnostics.values() for t in f.values() for c in t.values()]
    rotations=[c[k] for c in checks for k in ('fit_rotation','query_rotation') if k in c]
    metrics=[d['scaled_raw']['fit_metric'] for f in diagnostics.values() for d in f.values()]
    valid=[r for r in metrics if r['nonzero_pairs']]
    if len(data)!=original['summary']['query_occurrences']: raise ValueError('Incomplete grid')
    summary=dict(query_occurrences=len(data),unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient','target']].drop_duplicates()),
        full_screen={a:[t for t,r in results.items() if r['candidates'][a]['full_screen']] for a in CANDIDATES},
        nonconverged_fits=sum(not c['converged'] for c in checks),
        max_rotation_feature_error=max(c['feature_error'] for c in rotations),
        max_rotation_output_error=max(c['output_error'] for c in rotations),
        comparison_screens={a:{ref:[t for t,r in results.items() if r['candidates'][a]['comparisons'][ref]['screen']
                                  and r['all_solvers_converged']] for ref in (*COMMON,a.replace('scaled_','kernel_'))} for a in CANDIDATES},
        normalized_target_macro={l:{a:float(np.mean([r['normalized_mae'][l][a] for r in results.values()]))
                                   for a in ALL_ARMS} for l in ('patient','component')},
        metric=dict(fold_targets=len(metrics),median_numeric_fraction=float(np.median([r['numeric_fraction_median'] for r in valid])),
                    median_lag_fraction=float(np.median([r['median_fraction']['lag'] for r in valid])),
                    lag_dominant_above_90_percent=sum(r['median_fraction']['lag']>.9 for r in valid),
                    median_kernel_change_removing_numeric=float(np.median([r['numeric_removal_kernel_change_mean'] for r in valid]))))
    out.write_text(json.dumps(dict(summary=summary,targets=results,diagnostics=diagnostics,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Reused source only; diagonal fit scaling, not original-coordinate rotation invariance. Fixed finite kernel may underfit. No external/forecast/interval/promotion.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

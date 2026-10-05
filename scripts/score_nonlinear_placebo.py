"""Complete-family paired assessment of the matched body-readout placebo."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_additional_reconstruction import FAMILY_SIZE, paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.score_nonlinear_probe import ARMS
from scripts.train_nonlinear_placebo import SEED
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256


def assess(frame, seed, repetitions=5000):
    arms = [*ARMS, 'trained_shuffle']
    if (frame.empty or frame.target.nunique() != 1 or frame.role.nunique() != 1
            or frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *arms]]).all().all()):
        raise ValueError('Invalid paired rows')
    errors = frame[arms].subtract(frame.truth, axis=0).abs()
    errors['patient'] = frame.patient.to_numpy()
    patient = errors.groupby('patient', sort=True)[arms].mean()
    groups = frame.drop_duplicates('patient').set_index('patient').component
    patient['component'] = groups.reindex(patient.index)
    component = patient.groupby('component')[arms].mean()
    paired = dict(patient=paired_summary(patient.body_probe-patient.trained_shuffle,seed,repetitions),
        component=paired_summary(component.body_probe-component.trained_shuffle,seed+100,repetitions))
    passed = len(patient)>=30 and len(component)>=3 and all(
        p['mean']<0 and p['p_less'] is not None and p['p_less']<.05/FAMILY_SIZE for p in paired.values())
    return dict(query_occurrences=len(frame), unique_patients=len(patient), components=len(component),
        normalized_mae=dict(patient=patient[arms].mean().to_dict(), component=component.mean().to_dict()),
        body_minus_trained_shuffle=paired, paired_47_screen=bool(passed),
        per_component=[dict(component=str(c),patients=int((patient.component==c).sum()),
                            normalized_mae=row.to_dict()) for c,row in component.iterrows()])


def verify_rows(frame, prior, role, fold):
    keys = ['patient','target']
    if (frame.duplicated(keys).any() or prior.duplicated(keys).any()
            or not frame.role.eq(role).all() or not frame.fold.eq(fold).all()):
        raise ValueError('Duplicate/wrong role or fold')
    a, b = [f.set_index(keys).sort_index() for f in (frame,prior)]
    if (not a.index.equals(b.index) or not a.component.equals(b.component)
            or not np.isfinite(a.trained_shuffle).all()):
        raise ValueError('Changed support/provenance')
    if not np.allclose(a[['truth',*ARMS]],b[['truth',*ARMS]],rtol=1e-12,atol=1e-12):
        raise ValueError('Changed original predictions or truth')


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921/nonlinear_probe_20261001')
    output=Path('artifacts/native_reconstruction_nonlinear_placebo_assessment_20261003.json')
    if output.exists(): raise ValueError('Existing complete assessment')
    paths=[Path(__file__),Path('scripts/train_nonlinear_placebo.py'),
        Path('scripts/score_additional_reconstruction.py'),Path('scripts/score_nonlinear_probe.py'),
        Path('scripts/score_monitored_continuation_outer.py'),
        Path('RECONSTRUCTION_NONLINEAR_PLACEBO_PROTOCOL_20261003.md')]
    previous={}
    checked={}
    for role,suffix in (('selection','source_assessment_20261001'),('test','outer_assessment_20261002')):
        path=Path(f'artifacts/native_reconstruction_nonlinear_{suffix}.json')
        previous[role]=json.loads(path.read_text())
        verify_bound_files(previous[role]['bindings'],checked)
        paths.append(path)
    targets=sorted(previous['selection']['targets'])
    if len(targets)!=47 or targets!=sorted(previous['test']['targets']):
        raise ValueError('Changed target family')
    frames=[]; fixed={}
    for fold in range(5):
        rp=Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
        report=json.loads(rp.read_text()); verify_bound_files(report['bindings'],checked)
        if (report['fold']!=fold or report['seed']!=SEED or report['parameters']!=PARAMETERS
                or sorted(report['targets'])!=targets or report['promotion_allowed'] is not False):
            raise ValueError('Changed experiment')
        folder=root/f'placebo_20261003/fold{fold}'
        cp=folder/'contract.json'
        if json.loads(cp.read_text())!={k:report[k] for k in ('fold','parameters','seed','bindings')}:
            raise ValueError('Changed fitted contract')
        paths.extend((rp,cp))
        outer=pd.read_parquet(root/f'outer_20261002/fold{fold}.parquet')
        rows=[]; fixed[str(fold)]={}
        for target in targets:
            pp,mp=folder/f'{target}.parquet',folder/f'{target}.joblib'
            m=report['targets'][target]
            if (m['target']!=target or m['fold']!=fold or m['prediction_sha256']!=sha256(pp)
                    or m['model_sha256']!=sha256(mp) or m['contract_sha256']!=sha256(cp)):
                raise ValueError('Changed fitted artifact')
            frame=pd.read_parquet(pp)
            if (frame.groupby('role').size().to_dict()!=m['query_counts']
                    or set(frame.role)-{'selection','test'} or not frame.target.eq(target).all()):
                raise ValueError('Changed query counts')
            for role in ('selection','test'):
                prior=(pd.read_parquet(root/f'fold{fold}_probes/{target}.parquet') if role=='selection'
                       else outer.loc[outer.target==target])
                verify_rows(frame.loc[frame.role==role],prior,role,fold)
            rows.append(frame); paths.extend((pp,mp))
            fixed[str(fold)][target]=dict(fit_patients=m['fit_patients'],unchanged_singletons=m['unchanged_singletons'])
        frames.extend(rows)
        print('Verified placebo fold',fold,flush=True)
    data=pd.concat(frames,ignore_index=True)
    test=data.loc[data.role=='test']
    if (test.duplicated(['patient','target']).any() or test.groupby('patient').fold.nunique().max()!=1
            or test.groupby('component').fold.nunique().max()!=1):
        raise ValueError('Outer reuse')
    results={}; summaries={}
    for role in ('selection','test'):
        part=data.loc[data.role==role]; original=previous[role]
        expected=original['summary']['query_occurrences' if role=='selection' else 'pairs']
        if len(part)!=expected: raise ValueError('Incomplete full family')
        results[role]={}
        for i,target in enumerate(targets):
            v=assess(part.loc[part.target==target],SEED+(0 if role=='selection' else 100000)+1000*i)
            for level in ('patient','component'):
                for arm in ARMS:
                    if not np.isclose(v['normalized_mae'][level][arm],original['targets'][target]['normalized_mae'][level][arm],rtol=1e-10,atol=1e-10):
                        raise ValueError('Original aggregate mismatch')
            results[role][target]=v
            print('Scored placebo',role,i+1,'of47',flush=True)
        passes=[t for t,v in results[role].items() if v['paired_47_screen']]
        summaries[role]=dict(occurrences=len(part),unique_patients=int(part.patient.nunique()),
            normalized_target_macro={k:{a:float(np.mean([v['normalized_mae'][k][a] for v in results[role].values()]))
                for a in (*ARMS,'trained_shuffle')} for k in ('patient','component')},
            matched_placebo_screen=passes,
            prior_incremental_and_controls_and_placebo=sorted(set(passes).intersection(*[
                set(original['summary']['all_contrasts_screen_passes']['body_probe_minus_'+a])
                for a in ('native','process_summary','population')])))
    summaries['source_outer_placebo_overlap']=sorted(set(summaries['selection']['matched_placebo_screen']) & set(summaries['test']['matched_placebo_screen']))
    output.write_text(json.dumps(dict(summary=summaries,results=results,fit_reassignment=fixed,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Fixed within-component shuffle, one trained readout-null realization. Real-trained encoder retained. Conditional/reused-development screens, not independent source/outer replication, forecast/interval/runtime promotion or proof of physiology-only information.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()

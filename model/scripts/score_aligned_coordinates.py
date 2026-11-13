"""Source-only paired test of observation-aligned shared-state coordinates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_aligned_coordinates import ARMS, DEST, PARAMETERS, PRIVATE, ROOT, ALPHA
from whole_body_jepa.multisource_intervals import sha256

BASE = ('body_probe','native','raw_probe','process_summary','population')
REFS = (*BASE,'random_axes','shuffled_axes')
ALL = (*ARMS,*BASE)
FAMILY = 47*len(REFS)


def assess(frame,seed,repetitions=5000):
    if (frame.empty or frame.target.nunique() != 1 or frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth',*ALL]].to_numpy()).all()):
        raise ValueError('Invalid aligned coordinate query')
    errors=frame[list(ALL)].subtract(frame.truth,axis=0).abs()
    errors['patient']=frame.patient.to_numpy()
    people=errors.groupby('patient')[list(ALL)].mean()
    people['component']=frame.drop_duplicates('patient').set_index('patient').component.reindex(people.index)
    groups=people.groupby('component')[list(ALL)].mean()
    comparisons={}
    for i,ref in enumerate(REFS):
        paired={level:paired_summary(rows.aligned_body-rows[ref],seed+100*i+j,repetitions)
                for j,(level,rows) in enumerate((('patient',people),('component',groups)))}
        passed=len(people)>=30 and len(groups)>=3 and all(
            s['mean']<0 and s['p_less'] is not None and s['p_less']<.05/FAMILY
            and s['ci95'] is not None and s['ci95'][1]<0 for s in paired.values())
        comparisons[ref]=dict(paired=paired,screen=bool(passed),
            component_mean_harms=int((groups.aligned_body>groups[ref]).sum()))
    return dict(query_occurrences=len(frame),unique_patients=len(people),components=len(groups),
        normalized_mae=dict(patient=people[list(ALL)].mean().to_dict(),component=groups.mean().to_dict()),
        comparisons=comparisons,full_screen=all(c['screen'] for c in comparisons.values()),
        per_component=[dict(component=str(c),patients=int((people.component==c).sum()),normalized_mae=row.to_dict())
                       for c,row in groups.iterrows()])


def main():
    out=Path('artifacts/native_reconstruction_aligned_coordinates_assessment_20261003.json')
    if out.exists(): raise ValueError('Existing completed assessment')
    op=Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    original=json.loads(op.read_text());checked={};verify_bound_files(original['bindings'],checked)
    targets=sorted(original['targets'])
    if len(targets)!=47: raise ValueError('Incomplete target family')
    fp=ROOT/'partitions_20260922.pt';roles=torch.load(fp,weights_only=False)['folds']
    paths=[Path(__file__),Path('scripts/score_additional_reconstruction.py'),op,fp]
    frames=[];fits={};retention={}
    for f in range(5):
        p=Path(f'artifacts/native_reconstruction_aligned_coordinates_fold{f}_20261003.json')
        report=json.loads(p.read_text());verify_bound_files(report['bindings'],checked)
        cp=DEST/f'fold{f}/contract.json'
        if (report['fold']!=f or report['parameters']!=PARAMETERS or report['alpha']!=ALPHA
                or sorted(report['targets'])!=targets or json.loads(cp.read_text())['bindings']!=report['bindings']):
            raise ValueError('Changed fold contract')
        paths.extend((p,cp));fits[str(f)]={};retention[str(f)]={}
        for t,meta in report['targets'].items():
            qp,mp=[DEST/f'fold{f}/{t}.{s}' for s in ('parquet','joblib')]
            if (meta['prediction_sha256']!=sha256(qp) or meta['model_sha256']!=sha256(mp)
                    or meta['contract_sha256']!=sha256(cp)):
                raise ValueError('Changed saved artifact')
            frame=pd.read_parquet(qp);old=pd.read_parquet(PRIVATE/f'fold{f}_probes/{t}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient)!=set(old.patient)
                    or any(roles[f][i]!='selection' for i in frame.patient)
                    or not (frame.fold==f).all() or not (frame.target==t).all()
                    or len(frame)!=meta['selection_patients']):
                raise ValueError('Changed query identity')
            old=old.set_index('patient').loc[frame.patient]
            for c in ('truth','component',*BASE):
                if not np.array_equal(frame[c],old[c]): raise ValueError('Changed original reference')
            frames.append(frame);paths.extend((qp,mp))
            fits[str(f)][t]=meta['checks'];retention[str(f)][t]=meta['retention']
    data=pd.concat(frames,ignore_index=True);results={}
    for i,t in enumerate(targets):
        r=assess(data.loc[data.target==t],20261003+1000*i)
        for level in ('patient','component'):
            for arm in BASE:
                if not np.isclose(r['normalized_mae'][level][arm],original['targets'][t]['normalized_mae'][level][arm],atol=1e-12,rtol=1e-12):
                    raise ValueError('All-fold baseline mismatch')
        results[t]=r;print('Scored aligned coordinates',i+1,'of47',flush=True)
    if len(data)!=original['summary']['query_occurrences']: raise ValueError('Incomplete source grid')
    summary=dict(query_occurrences=len(data),unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient','target']].drop_duplicates()),
        full_screen=[t for t,r in results.items() if r['full_screen']],
        comparison_screens={ref:[t for t,r in results.items() if r['comparisons'][ref]['screen']] for ref in REFS},
        normalized_target_macro={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
                                       for a in ALL} for level in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,targets=results,fit_checks=fits,retention=retention,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Reused source development, same frozen body with extra deterministic axes. Extra split directions controlled by equal-width random and shuffled mappings. Not new query information, latent physiological truth, irreversible loss proof, forecast, independent folds or external promotion.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

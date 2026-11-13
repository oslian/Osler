"""Predeclared component-equal additive offsets, fitted on calibration only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.infer_additional_calibration import verify_roles
from scripts.score_additional_reconstruction import paired_summary, verify_frame, ARMS, FAMILY_SIZE
from whole_body_jepa.multisource_intervals import sha256


def target_union(frames):
    return sorted(set().union(*(set(f.target) for f in frames)))


def fit_offsets(frame):
    if frame.empty:
        return dict(status='insufficient_support',patients=0,components=0,supported_components=0,
                    offsets=dict.fromkeys(ARMS,0.))
    if frame.patient.duplicated().any() or not np.isfinite(frame[['truth',*ARMS]].to_numpy()).all():
        raise ValueError('Invalid calibration rows')
    sizes=frame.groupby('component').size()
    result=dict(patients=len(frame),components=len(sizes),supported_components=int((sizes>=30).sum()))
    if result['supported_components']<3:
        return dict(result,status='insufficient_support',offsets=dict.fromkeys(ARMS,0.))
    weights=(1/frame.component.map(sizes)).to_numpy(dtype=np.longdouble)
    offsets={}
    for arm in ARMS:
        residual=(frame.truth-frame[arm]).to_numpy()
        order=np.argsort(residual,kind='stable');w=weights[order]
        cdf=np.cumsum(w,dtype=np.longdouble)
        # Lower median at numerical ties, accounting only for summation precision.
        tolerance=np.finfo(np.longdouble).eps*len(w)*cdf[-1]
        index=np.searchsorted(cdf,cdf[-1]/2-tolerance,side='left')
        offsets[arm]=float(residual[order[index]])
    return dict(result,status='fitted',offsets=offsets)


def compare(frame, seed, repetitions=5000):
    names=('real','process','population','uncorrected_real')
    err=frame[list(names)].subtract(frame.truth,axis=0).abs()
    norm=err.div(frame.fit_scale,axis=0);group=err.groupby(frame.component).mean()
    comparisons={}
    for i,a in enumerate(names[1:]):
        comparisons[a]=dict(patient=paired_summary(err.real-err[a],seed+i,repetitions),
            component=paired_summary(group.real-group[a],seed+100+i,repetitions))
    passed=len(frame)>=30 and len(group)>=3 and all(
        s['mean']<0 and s['p_less'] is not None and s['p_less']<.05/FAMILY_SIZE
        for c in comparisons.values() for s in c.values())
    return dict(patients=len(frame),components=len(group),
        raw_mae=dict(patient=err.mean().to_dict(),component=group.mean().to_dict()),
        normalized_mae=dict(patient=norm.mean().to_dict(),component=norm.groupby(frame.component).mean().mean().to_dict()),
        comparisons=comparisons,incremental_descriptive_screen=bool(passed),
        per_component=[dict(component=str(c),patients=int((frame.component==c).sum()),raw_mae=row.to_dict())
                       for c,row in group.iterrows()])


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    out=Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    if out.exists():raise ValueError('Existing calibration assessment')
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt';pp=root/'provenance.parquet'
    roles=torch.load(fp,weights_only=False)['folds'];scaling=torch.load(sp,weights_only=False)
    links=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    components=dict(links.itertuples(index=False,name=None))
    paths=[Path(__file__),fp,sp,pp,Path('scripts/infer_additional_calibration.py'),
        Path('scripts/score_additional_reconstruction.py'),Path('RECONSTRUCTION_ADDITIONAL_CALIBRATION_PROTOCOL_20260924.md')]
    predictions={};checked={}
    # Validate all five calibration and outer files BEFORE fitting any offset.
    for fold in range(5):
        for kind in ('calibration','outer'):
            p=root/f'{kind}_20260924/fold{fold}.parquet';mp=p.with_suffix('.json')
            meta=json.loads(mp.read_text());paths.extend((p,mp))
            if sha256(p)!=meta['prediction_sha256'] or meta['fold']!=fold:raise ValueError('Changed predictions')
            for name,h in meta['bindings'].items():
                if name not in checked:checked[name]=sha256(Path(name))
                if checked[name]!=h:raise ValueError('Changed dependency')
            frame=pd.read_parquet(p)
            if kind=='calibration':
                if meta['role']!='calibration':raise ValueError('Wrong role')
                verify_roles(frame,scaling,roles[fold],fold,components)
            else:
                verify_frame(frame,scaling,roles[fold],fold)
                if not np.array_equal(frame.component.to_numpy(),frame.patient.map(components).to_numpy()):raise ValueError('Wrong provenance')
            predictions[fold,kind]=frame
    targets=target_union([predictions[f,'outer'] for f in range(5)])
    if len(targets)!=FAMILY_SIZE:raise ValueError('Changed family')
    fits={};corrected=[]
    for fold in range(5):
        cal=predictions[fold,'calibration'];test=predictions[fold,'outer']
        if set(cal.patient)&set(test.patient) or set(cal.component)&set(test.component):raise ValueError('Calibration-test overlap')
        fitted={};part=test.copy();part['uncorrected_real']=part.real
        for t in targets:
            fitted[t]=fit_offsets(cal.loc[cal.target==t])
            for a in ARMS:part.loc[part.target==t,a]+=fitted[t]['offsets'][a]
        fits[str(fold)]=fitted;corrected.append(part)
    data=pd.concat(corrected,ignore_index=True)
    if data.duplicated(['patient','target']).any():raise ValueError('Duplicated outer rows')
    results={}
    for i,t in enumerate(targets):
        results[t]=compare(data.loc[data.target==t],20260925+1000*i)
        print('Scored calibrated target',i+1,'of',FAMILY_SIZE,flush=True)
    summary=dict(pairs=len(data),patients=int(data.patient.nunique()),targets=len(targets),
        fitted_target_folds=sum(x['status']=='fitted' for f in fits.values() for x in f.values()),
        unsupported_target_folds=sum(x['status']!='fitted' for f in fits.values() for x in f.values()),
        incremental_descriptive_passes=[t for t,r in results.items() if r['incremental_descriptive_screen']],
        improved_patient_means=[t for t,r in results.items() if r['comparisons']['uncorrected_real']['patient']['mean']<0],
        improved_component_means=[t for t,r in results.items() if r['comparisons']['uncorrected_real']['component']['mean']<0],
        normalized_target_macro={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
            for a in ('real','process','population','uncorrected_real')} for level in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,offsets=fits,targets=results,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Point-offset follow-up using preassigned calibration roles only. Outer outcomes already viewed; reused validation, not untouched confirmation. Current masked reconstruction, not future forecast or conformal uncertainty. No slope or hyperparameter selection.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

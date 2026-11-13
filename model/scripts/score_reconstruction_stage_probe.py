"""Complete five-fold diagnostic assessment, with paired patient losses."""
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import torch
from scripts.audit_reconstruction_stage_probe import ROOT, DEST, SOURCES, choose
from scripts.cache_nonlinear_probe import decode_features
from scripts.score_additional_reconstruction import paired_summary
from whole_body_jepa.multisource_intervals import sha256

ARMS = tuple(f'{family}_{source}' for family in ('hgb','ridge') for source in SOURCES)
CONTRASTS = tuple((f'{family}_{left}',f'{family}_{right}')
    for family in ('hgb','ridge') for left,right in
    (('event_mean','body'),('event_mean','process'),('event_mean','raw'),('event_last','body')))


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.duplicated(['patient','fold']).any()
            or frame.target.nunique()!=1 or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth',*ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired stage predictions')
    err=frame[list(ARMS)].subtract(frame.truth,axis=0).abs()
    err['patient']=frame.patient.to_numpy()
    patients=err.groupby('patient')[list(ARMS)].mean()
    links=frame.drop_duplicates('patient').set_index('patient').component
    groups=patients.groupby(links.reindex(patients.index)).mean()
    comparisons={}
    for i,(left,right) in enumerate(CONTRASTS):
        paired=dict(patient=paired_summary(patients[left]-patients[right],seed+i,repetitions),
                    component=paired_summary(groups[left]-groups[right],seed+100+i,repetitions))
        passed=len(patients)>=30 and len(groups)>=3 and all(
            s['ci95'] is not None and s['ci95'][1]<0 and s['p_less'] is not None
            and s['p_less']<.05/(47*8) for s in paired.values())
        comparisons[left+'_minus_'+right]=dict(paired=paired,screen_pass=bool(passed))
    return dict(query_occurrences=len(frame),unique_patients=len(patients),components=len(groups),
        normalized_mae=dict(patient=patients.mean().to_dict(),component=groups.mean().to_dict()),
        comparisons=comparisons)


def main():
    output=Path('artifacts/native_reconstruction_stage_assessment_20261003.json')
    if output.exists(): raise ValueError('Existing stage assessment')
    roles=torch.load(ROOT/'partitions_20260922.pt',weights_only=False)['folds']
    scaling=torch.load(ROOT/'scaling_20260922.pt',weights_only=False)
    links=pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    components=dict(links.itertuples(index=False,name=None))
    checked={}; frames=[]; family=None
    for fold in range(5):
        rp=Path(f'artifacts/native_reconstruction_stage_fold{fold}_20261003.json')
        report=json.loads(rp.read_text()); contract=report['contract']
        cp=DEST/f'fold{fold}/contract.json'
        if (contract!=json.loads(cp.read_text()) or contract['fold']!=fold
                or len(report['targets'])!=47 or report['promotion_allowed'] is not False):
            raise ValueError('Wrong completed fold')
        if family is None: family=sorted(report['targets'])
        if sorted(report['targets'])!=family: raise ValueError('Changed family')
        for p,h in contract['bindings'].items():
            if p not in checked: checked[p]=sha256(Path(p))
            if checked[p]!=h: raise ValueError('Changed bound dependency: '+p)
        dbpath=ROOT/f'nonlinear_probe_20261001/fold{fold}_selection.sqlite'
        db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for target,saved in report['targets'].items():
                path=DEST/f'fold{fold}/{target}.parquet'
                if (saved['target']!=target or saved['fold']!=fold
                        or saved['prediction_sha256']!=sha256(path)
                        or saved['contract_sha256']!=sha256(cp)):
                    raise ValueError('Changed target output')
                frame=pd.read_parquet(path)
                c=scaling['variables'].index(target)
                eligible=[r[0] for r in db.execute('SELECT patient FROM features WHERE target=?',(c,))]
                ids=choose(eligible,256)
                if (frame.patient.tolist()!=ids or len(frame)!=saved['selection_patients']
                        or not (frame.target==target).all() or not (frame.fold==fold).all()
                        or any(roles[fold][p]!='selection' for p in ids)
                        or frame.component.tolist()!=[components[p] for p in ids]):
                    raise ValueError('Sample, role or provenance mismatch')
                for p,y in zip(ids,frame.truth):
                    row=db.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?',(p,c)).fetchone()
                    if float(decode_features(*row)['truth'])!=y: raise ValueError('Changed target truth')
                means=frame[list(ARMS)].subtract(frame.truth,axis=0).abs().mean()
                if any(not np.isclose(means[k],saved['normalized_mae'][k],atol=1e-12,rtol=1e-12) for k in ARMS):
                    raise ValueError('Changed reported scores')
                checked[str(path)]=sha256(path); frames.append(frame)
        finally: db.close()
        checked[str(cp)]=sha256(cp); checked[str(rp)]=sha256(rp)
        print('Verified stage fold',fold,flush=True)
    data=pd.concat(frames,ignore_index=True)
    results={t:assess(data[data.target==t],20261003+i*1000) for i,t in enumerate(family)}
    summary=dict(query_occurrences=len(data),unique_patients=int(data.patient.nunique()),
        normalized_target_macro={level:{arm:float(np.mean([r['normalized_mae'][level][arm] for r in results.values()]))
            for arm in ARMS} for level in ('patient','component')},
        screens={a+'_minus_'+b:[t for t,r in results.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
            for a,b in CONTRASTS})
    for p in (Path(__file__),Path('scripts/score_additional_reconstruction.py')): checked[str(p)]=sha256(p)
    output.write_text(json.dumps(dict(summary=summary,targets=results,bindings=checked,
        runtime_modified=False,promotion_allowed=False,
        limitation='Fixed bounded source diagnostic, reused development. Same512/256 hash sample per target/fold. Overlapping folds averaged per patient; conditional uncertainty. Not clinical, causal, future or new lockbox. Equal223 dimensions among latent arms do not establish all information retained or lost.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

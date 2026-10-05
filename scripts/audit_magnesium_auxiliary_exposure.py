"""Replay actual auxiliary sample coefficients, not gradient magnitudes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/magnesium_auxiliary_20260911.json')
    report=json.loads(parent.read_text())
    for name,digest in report['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_auxiliary_exposure_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    lab=dict(np.load(root/'multitarget_labels.npz'));variables=list(lab['variables']);k=variables.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    eligible=lab['paired'][:,k]&lab['paired'][:,variables.index('glucose')]
    items=[]
    for entry in report['seeds']:
        seed=entry['seed'];fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]&eligible
        pools=[np.flatnonzero(fit&q.uniquepid.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'uniquepid']))]
        supported=np.array([q.loc[fit&lab['paired'][:,j],'uniquepid'].nunique()>=30 for j in range(len(variables))])
        supported[k]=False
        active=lab['paired']&supported[None,:]
        rng=np.random.default_rng(seed);coeff=np.zeros_like(active,dtype=float)
        selected=entry['arms']['real']['selected_epoch'];steps=selected*16
        for _ in range(steps):
            chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
            a=active[chosen];counts=a.sum(axis=0);targets=int((counts>0).sum())
            for j in np.flatnonzero(counts):
                np.add.at(coeff[:,j],chosen[a[:,j]],.25/(targets*counts[j]))
        table={}
        for j in np.flatnonzero(supported):
            mask=fit&lab['paired'][:,j]
            mass=pd.Series(coeff[mask,j],index=q.loc[mask,'uniquepid']).groupby(level=0).sum().to_numpy()
            table[variables[j]]=dict(eligible_patients=len(mass),seen_patients=int((mass>0).sum()),
                mean_loss_coefficient=float(mass.sum()/max(steps,1)),
                effective_patients=float(mass.sum()**2/np.square(mass).sum()) if mass.sum() else 0.)
        items.append(dict(seed=seed,selected_epoch=selected,primary_loss_coefficient=1.,targets=table))
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        coefficient_not_gradient=True,seeds=items,binding={str(p):sha256(p) for p in [parent,Path(__file__),Path('scripts/train_auxiliary_encoder.py')]}),indent=2)+'\n')
    common=set.intersection(*(set(s['targets']) for s in items))
    for name in sorted(common):
        records=[s['targets'][name] for s in items]
        print(name,'coefficient',round(float(np.median([r['mean_loss_coefficient'] for r in records])),5),
              'patient effective fraction',round(float(np.median([r['effective_patients']/r['eligible_patients'] for r in records])),3),
              'coverage',min(r['seen_patients']/r['eligible_patients'] for r in records))


if __name__=='__main__':main()

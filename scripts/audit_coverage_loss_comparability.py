"""Replay exact training draws to expose changing loss evaluation populations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.bounded_site_sampling import bounded_weights
from scripts.coverage_sampling import coverage_indices
from whole_body_jepa.multisource_intervals import sha256


def smooth_l1_numpy(delta):
    a=np.abs(np.asarray(delta,float))
    return np.where(a<1,.5*a*a,a-.5)


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    s=torch.load(root/'scaling_20260922.pt',weights_only=False)
    folds=torch.load(root/'partitions_20260922.pt',weights_only=False)['folds']
    links=pd.read_parquet(root/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    components=dict(links.itertuples(index=False,name=None))
    results={};bindings={}
    for fold in (0,1,3,4):
        path=root/f'coverage_training_20260925/fold{fold}_real.pt'
        ck=torch.load(path,weights_only=False);bindings[str(path)]=sha256(path)
        scaler=s['scalers'][fold];fit=np.array([folds[fold][p]=='fit' for p in s['people']])
        pools={int(c):np.flatnonzero(fit & np.isfinite(s['centers'][:,c])) for c in np.flatnonzero(scaler['counts'])}
        weights={c:bounded_weights([components[s['people'][i]] for i in ids]) for c,ids in pools.items()}
        rng=np.random.default_rng(19);pending={c:np.array([],dtype=np.int64) for c in pools}
        seen=set();epochs=[]
        for epoch in range(8):
            schedule=[];losses=[]
            for c,ids in pools.items():
                draws,pending[c]=coverage_indices(weights[c],rng,1072,pending[c])
                values=(s['centers'][ids[draws],c]-scaler['location'][c])/scaler['scale'][c]
                losses.extend(smooth_l1_numpy(values-ck['medians'][c]).tolist())
                schedule.extend((int(ids[i]),c) for i in draws)
            rng.shuffle(schedule);seen.update(schedule)
            if len(seen)!=ck['log'][epoch]['unique_fit_pairs_seen']:raise ValueError('Wrong replay')
            base=float(np.mean(losses));real=ck['log'][epoch]['fit_draw_loss']
            epochs.append(dict(epoch=epoch+1,constant_median_draw_loss=base,online_training_draw_loss=real,
                               baseline_minus_training_loss=base-real))
        if rng.bit_generator.state!=ck['rng'] or seen!=ck['seen']:raise ValueError('Replay differs from checkpoint')
        for c in pending:np.testing.assert_array_equal(pending[c],ck['pending'][c])
        b=np.array([e['constant_median_draw_loss'] for e in epochs]);r=np.array([e['online_training_draw_loss'] for e in epochs])
        results[str(fold)]=dict(epochs=epochs,constant_loss_range=float(np.ptp(b)),
            training_loss_range=float(np.ptp(r)),draw_difficulty_correlation=float(np.corrcoef(b,r)[0,1]))
    output=dict(folds=results,bindings=bindings,promotion_allowed=False,runtime_modified=False,
        limitation='Fit-only exact draw replay. Constant predictor loss variation is sampling-composition drift, not learning. Correlation across8epochs is descriptive. Online loss uses changing weights, so baseline-relative loss still cannot certify convergence.')
    Path('artifacts/native_reconstruction_coverage_loss_comparability_20260929.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

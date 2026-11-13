"""Certify identical fit problem/schedule for fixed folds 0 and 2 only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.bounded_site_sampling import bounded_weights, sample_indices
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    out=Path('artifacts/native_reconstruction_additional_fold_reuse_20260923.json')
    if out.exists():raise ValueError('Existing certificate')
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt';pp=root/'provenance.parquet'
    folds=torch.load(fp,weights_only=False)['folds'];s=torch.load(sp,weights_only=False)
    people=s['people'];centers=s['centers'];variables=s['variables']
    assert {p for p,r in folds[0].items() if r=='fit'}=={p for p,r in folds[2].items() if r=='fit'}
    for key in s['scalers'][0]:np.testing.assert_array_equal(s['scalers'][0][key],s['scalers'][2][key])
    provenance=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    assert not provenance.uniquepid.duplicated().any()
    component=dict(provenance.itertuples(index=False,name=None))
    plans=[];baselines=[];states=[];seen_counts=[]
    for fold in (0,2):
        scaler=s['scalers'][fold];fit=np.array([folds[fold][p]=='fit' for p in people])
        pools={int(c):np.flatnonzero(fit & np.isfinite(centers[:,c])) for c in np.flatnonzero(scaler['counts'])}
        weights={c:bounded_weights([component[people[i]] for i in ids]) for c,ids in pools.items()}
        baselines.append({c:float(np.median((centers[ids,c]-scaler['location'][c])/scaler['scale'][c])) for c,ids in pools.items()})
        rng=np.random.default_rng(19);digests=[];seen=set();counts=[]
        for epoch in range(8):
            schedule=[]
            for c,ids in pools.items():schedule.extend((int(ids[i]),c) for i in sample_indices(weights[c],rng,1072))
            rng.shuffle(schedule);seen.update(schedule);counts.append(len(seen))
            digests.append(hashlib.sha256(np.asarray(schedule,dtype='<i8').tobytes()).hexdigest())
        plans.append(digests);states.append(rng.bit_generator.state);seen_counts.append(counts)
    assert plans[0]==plans[1] and baselines[0]==baselines[1] and states[0]==states[1]
    training=root/'training_20260923';checkpoints={};sources={}
    cache=root/'full_histories_20260922.sqlite'
    cache_report=json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    assert sha256(cache)==cache_report['private_sha256']
    for arm in ('real','process'):
        dest=training/f'fold2_{arm}.pt'
        if dest.exists():raise ValueError('Existing destination weights')
        source=training/f'fold0_{arm}.pt';ck=torch.load(source,weights_only=False)
        assert ck['epoch']==8 and ck['fold']==0 and ck['arm']==arm
        assert ck['medians']==baselines[0] and ck['rng']==states[0]
        assert [x['unique_fit_pairs_seen'] for x in ck['log']]==seen_counts[0]
        assert all(x['updates']==50384 for x in ck['log'])
        assert all(sha256(Path(p))==h for p,h in ck['bindings'].items())
        assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
        sources[arm]=dict(path=str(source),sha256=sha256(source))
        checkpoints[arm]=deepcopy(ck)
    bindings={str(p):sha256(p) for p in (Path(__file__),fp,sp,pp,Path('scripts/train_additional_reconstruction.py'),Path('scripts/bounded_site_sampling.py'))}
    destinations={}
    for arm,ck in checkpoints.items():
        ck['fold']=2;ck['reuse_certificate']=str(out);ck['reused_from']=sources[arm]
        dest=training/f'fold2_{arm}.pt';torch.save(ck,dest)
        check=torch.load(dest,weights_only=False)
        assert all(torch.equal(check['state_dict'][k],ck['state_dict'][k]) for k in ck['state_dict'])
        destinations[arm]=dict(path=str(dest),sha256=sha256(dest))
    result=dict(source_fold=0,destination_fold=2,fit_identity_equal=True,scalers_equal=True,
        all_eight_ordered_schedule_hashes=plans[0],baselines_equal=True,final_rng_equal=True,
        unique_pair_counts=seen_counts[0],sources=sources,destinations=destinations,bindings=bindings,
        independently_trained=False,completed_equivalent_stages=16,runtime_modified=False,promotion_allowed=False,
        rationale='Training code uses fold only for fit membership, scaler, provenance and output naming. Exact fit inputs/scalers, weighted schedule, architecture, seed and fixed optimization settings are identical. Shared cache hash and all source checkpoint dependencies verified. Test/calibration patients do not enter fitting. This is equivalent computation, not a new independent training replication.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('bindings','sources','destinations')},indent=2),flush=True)


if __name__=='__main__':main()

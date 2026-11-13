"""Freeze outcome-blind hospital and patient roles for expanded diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.multisource_intervals import sha256


def held_hospital(value):
    return int.from_bytes(hashlib.sha256(f'insulin-potassium-hospital-v1:{value}'.encode()).digest()[:8],'big')%5==0


def split_roles(frame, seed):
    hospital=frame.hospitalid.map(held_hospital).to_numpy(bool)
    external_subjects=set(frame.loc[hospital,'subject_id'])
    eligible=~frame.subject_id.isin(external_subjects).to_numpy()
    fit,tune,evaluation=three_way_partition(frame.subject_id,eligible,seed)
    return dict(fit=fit,tune=tune,evaluation=evaluation,hospital=hospital,
                excluded=~(fit|tune|evaluation|hospital))


def main():
    report_path=Path('artifacts/insulin_potassium_partition_20260911.json')
    if report_path.exists():raise ValueError('fresh output required')
    private=Path('/private/tmp/insulin_potassium_pairs_20260911')
    prior=json.loads(Path('artifacts/insulin_potassium_pairs_20260911.json').read_text())
    for name in ('queries.parquet','labels.npz'):
        path=private/name
        if sha256(path)!=prior['private_cache_hashes'][str(path)]:raise ValueError('cache changed')
    q=pd.read_parquet(private/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    y=np.load(private/'labels.npz');valid=y['paired'][:,0];q=q.loc[valid].reset_index(drop=True)
    sequence=y['sequence'][valid]
    if q[['subject_id','hospitalid']].isna().any().any():raise ValueError('missing identity')
    report=dict(complete=True,outcome_blind=True,pristine_lockbox=False,minimum_evaluation_patients=30,seeds=[])
    for seed in (7,19,31,43,59,71,83):
        roles=split_roles(q,seed)
        patient_sets={k:set(q.loc[m,'subject_id']) for k,m in roles.items() if k!='excluded'}
        keys=list(patient_sets)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if patient_sets[a]&patient_sets[b]:raise ValueError('patient leakage')
        stats={k:dict(rows=int(m.sum()),patients=int(q.loc[m,'subject_id'].nunique()),
                      sequence_patients=int(q.loc[m&sequence,'subject_id'].nunique()),hospitals=int(q.loc[m,'hospitalid'].nunique())) for k,m in roles.items()}
        path=private/f'partition_{seed}.npz';np.savez_compressed(path,**roles)
        report['seeds'].append(dict(seed=seed,roles=stats,partition_sha256=sha256(path),
            sequence_evaluation_support=all(stats[k]['sequence_patients']>=30 for k in ('evaluation','hospital'))))
    report['binding']={str(p):sha256(p) for p in (Path(__file__),private/'queries.parquet',private/'labels.npz')}
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

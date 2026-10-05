"""Descriptive error complementarity; outcome oracle is never deployable."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    output=Path('artifacts/clean_context_complementarity_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for p,h in ref['binding'].items():
        if sha256(Path(p))!=h:raise ValueError('parent changed')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    col=labels['variables'].tolist().index('glucose');q['truth']=labels['future'][:,col]
    result=dict(complete=True,runtime_modified=False,training_performed=False,
        status='descriptive_not_prospective_selector',seeds=[],
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        p=dict(np.load(root/f'conflict_repair_predictions_{seed}.npz'))
        own,full=p['clean_own'],p['clean_full'];blend=(own+full)/2
        oracle=np.where(np.abs(full-q.truth.to_numpy())<np.abs(own-q.truth.to_numpy()),full,own)
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m];truth=f.truth.to_numpy()
            errors=pd.DataFrame({'subject':f.subject_id.to_numpy(),'own':own[m]-truth,'full':full[m]-truth})
            by=errors.groupby('subject')[['own','full']].mean()
            absby=errors.assign(own=errors.own.abs(),full=errors.full.abs()).groupby('subject')[['own','full']].mean()
            item['roles'][role]=dict(patients=len(by),patient_mean_signed_error_correlation=float(by.own.corr(by.full)),
                fraction_patients_full_better=float((absby.full<absby.own).mean()),
                mae={name:patient_equal_mae(f,f.truth,x[m]) for name,x in [('own',own),('full',full),('equal_blend',blend),('outcome_oracle',oracle)]},
                equal_blend_vs_own=paired(f,blend[m],own[m],f'clean-blend-{seed}-{role}',1000),
                outcome_oracle_vs_own=paired(f,oracle[m],own[m],f'clean-oracle-{seed}-{role}',1000))
        result['seeds'].append(item)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        rs=[s['roles'][role] for s in result['seeds']]
        print(role,'correlation',float(np.median([r['patient_mean_signed_error_correlation'] for r in rs])),
              'fraction_better',float(np.median([r['fraction_patients_full_better'] for r in rs])),
              'mae',{a:float(np.median([r['mae'][a] for r in rs])) for a in rs[0]['mae']},
              'blend_gain',sum(r['equal_blend_vs_own']['ci95'][1]<0 for r in rs),
              'blend_harm',sum(r['equal_blend_vs_own']['ci95'][0]>0 for r in rs))


if __name__=='__main__':main()

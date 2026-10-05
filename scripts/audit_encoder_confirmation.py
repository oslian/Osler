"""Evaluate fixed seven-seed repair predictions without refitting or selection."""
from pathlib import Path
import argparse
import json
import duckdb
import numpy as np
import pandas as pd

from scripts.train_availability_encoder import CONFIRM_ARMS
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def metadata(role):
    with duckdb.connect() as con:
        if role == 'eicu_hospital':
            return con.execute("""SELECT 'eicu:' || CAST(patientunitstayid AS VARCHAR) AS stay_id,
                unittype AS careunit, CAST(hospitaldischargeyear AS VARCHAR) AS period
                FROM read_csv_auto('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')""").df()
        return con.execute("""SELECT 'mimic:' || CAST(i.stay_id AS VARCHAR) AS stay_id,
            i.first_careunit AS careunit, p.anchor_year_group AS period
            FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz') i
            JOIN read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/hosp/patients.csv.gz') p
            ON i.subject_id=p.subject_id""").df()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/encoder_confirm_20260910.json'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/encoder_confirm_strata_20260910.json'))
    args=parser.parse_args()
    source=args.source
    trained=json.loads(source.read_text())
    if not trained.get('complete'): raise ValueError('training incomplete')
    report=dict(status='candidate_only',promotion_allowed=False,runtime_modified=False,
                temporal_subgroups_not_temporal_holdout=True,untouched_lockbox=False,
                binding={str(source):sha256(source),__file__:sha256(Path(__file__)),
                    'ENCODER_CONFIRM_STRATA_PROTOCOL_20260910.md':sha256(Path('ENCODER_CONFIRM_STRATA_PROTOCOL_20260910.md'))},roles={})
    contrasts=[(f'{v}_unbounded',f'{v}_bounded') for v in ('own','whole')]
    contrasts += [(f'whole_{m}',f'own_{m}') for m in ('bounded','unbounded')]
    contrasts += [(a,'persistence') for a in CONFIRM_ARMS]
    for role in ('eicu_hospital','mimic'):
        path=Path(f'/private/tmp/availability_glucose_{role}.parquet')
        if sha256(path)!=trained['data_binding'][str(path)]: raise ValueError('row cache changed')
        f=pd.read_parquet(path)
        f=f.loc[f.fresh & f.truth.notna()].reset_index(drop=True)
        f=f.merge(metadata(role),on='stay_id',how='left',validate='many_to_one',sort=False)
        f[['careunit','period']]=f[['careunit','period']].fillna('unknown')
        groups={'all':np.ones(len(f),dtype=bool)}
        for column in ('careunit','period'):
            for value in sorted(f[column].unique()): groups[f'{column}:{value}']=f[column].eq(value).to_numpy()
        results={}
        for name,mask in groups.items():
            sub=f.loc[mask]
            results[name]=dict(rows=len(sub),patients=int(sub.subject_id.nunique()),seeds=[])
            supported=sub.subject_id.nunique()>=30 and not name.endswith(':unknown')
            results[name]['supported']=bool(supported)
            if not supported: continue
            for entry in trained['seeds']:
                seed=entry['seed']
                directory=Path(entry['training']['own_bounded']['checkpoint']).parent
                prediction_path=directory/f'predictions_{role}_{seed}.npz'
                with np.load(prediction_path,allow_pickle=False) as loaded:
                    preds={k:loaded[k][mask] for k in (*CONFIRM_ARMS,'persistence')}
                metrics={}
                actual=sub.truth.to_numpy()-sub.current.to_numpy()
                for arm,pred in preds.items():
                    delta=pred-sub.current.to_numpy()
                    direction=pd.DataFrame({'subject':sub.subject_id.to_numpy(),
                        'correct':np.sign(delta)==np.sign(actual)}).groupby('subject').correct.mean().mean()
                    corr=float(np.corrcoef(delta,actual)[0,1]) if delta.std()>0 and actual.std()>0 else None
                    metrics[arm]=dict(mae=patient_equal_mae(sub,sub.truth,pred),
                        patient_equal_direction=float(direction),row_delta_correlation=corr)
                results[name]['seeds'].append(dict(seed=seed,metrics=metrics,
                    prediction_sha256=sha256(prediction_path),comparisons={f'{a}_vs_{b}':
                        paired(sub,preds[a],preds[b],f'confirm:{role}:{name}:{seed}:{a}:{b}',1000)
                        for a,b in contrasts}))
        report['roles'][role]=results
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()

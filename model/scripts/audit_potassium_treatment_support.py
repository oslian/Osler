"""Aggregate infusion coverage on the fixed potassium diagnostic cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import classify_action
from extractors.eicu_aki_transition_extract import classify_aki_action
from osler_jepa.treatment_units import TARGET_TREATMENT_ACTIONS, ACTION_RATE_DIMENSIONS
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.multisource_intervals import sha256


def window_support(queries, events, hours=6.):
    """Event-time support only, not a guarantee of prospective availability."""
    result=np.zeros(len(queries),bool)
    groups={str(s):np.sort(g.offset.to_numpy(float)) for s,g in events.groupby('stay_id')}
    for i,row in enumerate(queries.itertuples()):
        values=groups.get(str(row.stay_id),np.array([]))
        result[i]=np.searchsorted(values,row.offset,side='right')>np.searchsorted(values,row.offset-hours*60,side='right')
    return result


def main():
    output=Path('artifacts/potassium_treatment_support_20260911_v2.json')
    if output.exists():raise ValueError('fresh output required')
    label=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    root=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    path=root/'infusionDrug.csv.gz'
    from whole_body_jepa.runtime import load_release
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    y=np.load(label);valid=y['paired'][:,col]
    queries=pd.DataFrame(dict(stay_id=y['stay_id'][valid].astype(str),subject_id=y['subject_id'][valid],
        offset=(pd.to_datetime(y['query_time'][valid])-pd.Timestamp('2100-01-01')).total_seconds()/60))
    numeric=queries.stay_id.str.replace('eicu:','',regex=False).astype(int)
    queries['stay_id']=numeric.astype(str);ids=set(numeric)
    configs=[get_body_system_config(n) for n in ('electrolyte_acid_base','endocrine_stress')]
    actions=set(TARGET_TREATMENT_ACTIONS['potassium']);records=[];cohort_rows=0
    columns=list(pd.read_csv(path,nrows=0).columns)
    for chunk in pd.read_csv(path,chunksize=500000,low_memory=False):
        chunk=chunk[chunk.patientunitstayid.isin(ids)];cohort_rows+=len(chunk)
        for row in chunk.itertuples(index=False):
            offset=pd.to_numeric(row.infusionoffset,errors='coerce')
            if not np.isfinite(offset):continue
            label_actions=set(classify_aki_action(row.drugname))
            for config in configs:label_actions.update(classify_action(config,row.drugname))
            for action in actions&label_actions:
                channels=rate_channels(dict(source='eicu_infusiondrug',rate_field='drugrate',rate_like=row.drugrate,original_label=row.drugname),action)
                records.append(dict(source_id=str(row.infusiondrugid),stay_id=str(row.patientunitstayid),
                    offset=float(offset),action=action,rate_known=bool(np.isfinite(channels['action_rate'])),
                    solution_flow_known=bool(np.isfinite(channels['solution_flow_ml_per_hour'])),
                    old_aki_classified=action in classify_aki_action(row.drugname)))
    events=pd.DataFrame(records,columns=['source_id','stay_id','offset','action','rate_known','solution_flow_known','old_aki_classified']).drop_duplicates(['source_id','action'])
    report=dict(status='coverage_only',complete=True,runtime_modified=False,promotion_allowed=False,
        external_data_read=False,queries=len(queries),patients=int(queries.subject_id.nunique()),
        cohort_infusion_rows=cohort_rows,infusion_columns=columns,
        independently_known_availability_column=False,actions={},
        binding={str(p):sha256(p) for p in (label,path,Path(__file__),Path('whole_body_jepa/event_treatment.py'))})
    for action in sorted(actions):
        e=events[events.action.eq(action)];q=window_support(queries,e);qr=window_support(queries,e[e.rate_known.astype(bool)])
        report['actions'][action]=dict(events=len(e),stays=int(e.stay_id.nunique()),
            solution_flow_known_events=int(e.solution_flow_known.sum()),
            six_hour_solution_flow_windows=int(window_support(queries,e[e.solution_flow_known.astype(bool)]).sum()),
            rate_known_events=int(e.rate_known.sum()),six_hour_event_windows=int(q.sum()),
            six_hour_rate_windows=int(qr.sum()),patients_with_event=int(queries.loc[q,'subject_id'].nunique()),
            omitted_by_aki_classifier=int((~e.old_aki_classified.astype(bool)).sum()))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['actions'],indent=2),flush=True)


if __name__=='__main__':main()

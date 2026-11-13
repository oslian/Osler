"""Check if rare treatment features can form valid tree leaves, source only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import classify_action
from extractors.eicu_aki_transition_extract import classify_aki_action
from whole_body_jepa.potassium_treatment_context import ACTIONS, action_support_status
from scripts.audit_potassium_treatment_support import window_support
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def synthetic_probe(min_leaf):
    x=np.zeros((200,1));x[:20]=1
    y=x[:,0].copy()
    model=HistGradientBoostingRegressor(loss='absolute_error', max_iter=200,
        min_samples_leaf=min_leaf, early_stopping=False, random_state=7).fit(x,y)
    prediction=model.predict(x)
    return dict(mae=float(np.abs(y-prediction).mean()),
                contrast=float(prediction[:20].mean()-prediction[20:].mean()))


def main():
    output=Path('artifacts/treatment_leaf_support_20260911_v2.json')
    if output.exists():raise ValueError('fresh output required')
    label=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    path=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    y=np.load(label);valid=y['paired'][:,col]
    q=pd.DataFrame(dict(stay_id=pd.Series(y['stay_id'][valid]).str.replace('eicu:','',regex=False),subject_id=y['subject_id'][valid],offset=(pd.to_datetime(y['query_time'][valid])-pd.Timestamp('2100-01-01')).total_seconds()/60))
    ids=set(q.stay_id.astype(int));records=[];configs=[get_body_system_config(n) for n in ('electrolyte_acid_base','endocrine_stress')]
    for chunk in pd.read_csv(path,chunksize=500000,low_memory=False):
        for row in chunk[chunk.patientunitstayid.isin(ids)].itertuples(index=False):
            actions=set(classify_aki_action(row.drugname))
            for config in configs:actions.update(classify_action(config,row.drugname))
            for action in set(ACTIONS)&actions:
                records.append(dict(stay_id=str(row.patientunitstayid),source_id=row.infusiondrugid,offset=row.infusionoffset,action=action))
    e=pd.DataFrame(records).drop_duplicates(['source_id','action'])
    supports={action:window_support(q,e[e.action.eq(action)]) for action in ACTIONS}
    support=supports['potassium_repletion']
    report=dict(complete=True,status='capacity_diagnostic_not_forecast_repair',runtime_modified=False,
                supported_windows=int(support.sum()),minimum_leaf_rows=30,seeds=[],
                synthetic={str(n):synthetic_probe(n) for n in (30,10)},
                binding={str(p):sha256(p) for p in (label,path,Path(__file__),Path('whole_body_jepa/potassium_treatment_context.py'))})
    for seed in (7,19,31,43,59,71,83):
        masks=three_way_partition(q.subject_id,np.ones(len(q),bool),seed)
        counts={name:int((m&support).sum()) for name,m in zip(('fit','tune','evaluation'),masks)}
        report['seeds'].append(dict(seed=seed,counts=counts,pure_exposed_leaf_possible=counts['fit']>=30,
            actions={action:action_support_status(q.subject_id,s,masks) for action,s in supports.items()}))
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()

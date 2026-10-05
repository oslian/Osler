"""Outcome-free inventory of documented same-label insulin rate sequences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.multisource_intervals import sha256


def read_insulin_rates(path):
    records=[]
    for chunk in pd.read_csv(path,chunksize=500000,low_memory=False):
        selected=chunk[chunk.drugname.fillna('').str.contains('insulin',case=False,regex=False)]
        for row in selected.itertuples(index=False):
            value=rate_channels(dict(source='eicu_infusiondrug',rate_field='drugrate',rate_like=row.drugrate,original_label=row.drugname),'insulin')['action_rate']
            offset=pd.to_numeric(row.infusionoffset,errors='coerce')
            if np.isfinite(value) and np.isfinite(offset):
                records.append((row.patientunitstayid,row.infusiondrugid,str(row.drugname),float(offset),value))
    return pd.DataFrame(records,columns=['stay_id','source_id','label','offset','value']).drop_duplicates('source_id')


def main():
    output=Path('artifacts/eicu_insulin_sequence_inventory_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    path=root/'infusionDrug.csv.gz'
    events=read_insulin_rates(path)
    stays=set();changing=set();pairs=0
    for (stay,label),g in events.groupby(['stay_id','label']):
        clean=g.groupby('offset').filter(lambda x:x.value.nunique()==1).drop_duplicates('offset').sort_values('offset')
        gap=clean.offset.diff();valid=(gap>0)&(gap<360)
        if valid.any():stays.add(stay)
        if (valid & clean.value.diff().abs().gt(1e-8)).any():changing.add(stay)
        pairs+=int(valid.sum())
    meta_path=root/'patient.csv.gz'
    meta=pd.read_csv(meta_path,usecols=['patientunitstayid','uniquepid','hospitalid'])
    label_path=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    y=np.load(label_path);col=list(y['variables']).index('potassium')
    old=set(pd.Series(y['stay_id'][y['paired'][:,col]]).str.replace('eicu:','',regex=False).astype(int))
    def count(ids):
        m=meta[meta.patientunitstayid.isin(ids)]
        return dict(stays=len(m),patients=int(m.uniquepid.nunique()),hospitals=int(m.hospitalid.nunique()))
    report=dict(complete=True,status='inventory_only_no_outcomes',runtime_modified=False,
                independent_availability=False,compatible_rate_events=len(events),adjacent_pairs=pairs,
                same_label_sequence=count(stays),changing_rate_sequence=count(changing),
                existing_diagnostic_stays_with_any_sequence=count(stays&old),
                outside_diagnostic_stays_with_any_sequence=count(stays-old),
                label_aligned_support_established=False,
                binding={str(p):sha256(p) for p in [path,meta_path,label_path,Path(__file__),Path('whole_body_jepa/event_treatment.py')]})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

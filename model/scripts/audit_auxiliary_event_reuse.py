"""Reproduce source labels and quantify shared future-event supervision."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256


def main():
    path=Path('/private/tmp/auxiliary_supervision_20260910.npz');old=np.load(path)
    q=pd.DataFrame(dict(stay_id=old['stay_id'],subject_id=old['subject_id'],t=old['query_time']))
    raw=pd.concat([pd.read_parquet('/private/tmp/availability_jepa_eicu_labs.parquet'),
        pd.read_parquet('/private/tmp/encoder_source_expansion_20260910/labs.parquet')],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(q.stay_id)]
    rebuilt=future_pairs(q,raw,list(old['variables']))
    for name in ('current','future','measurement_age','actual_delta_t','paired'):
        np.testing.assert_equal(old[name],rebuilt[name])
    report=dict(status='source_diagnostic_only',labels_exact=True,runtime_modified=False,
        binding={str(path):sha256(path),str(Path(__file__)):sha256(Path(__file__)),
                 'whole_body_jepa/availability_supervision.py':sha256(Path('whole_body_jepa/availability_supervision.py'))},targets={})
    targets=json.loads(Path('artifacts/auxiliary_supervision_20260910.json').read_text())['eligible_auxiliary_variables']+['glucose']
    for var in targets:
        col=list(old['variables']).index(var);mask=rebuilt['paired'][:,col]
        f=q.loc[mask,['subject_id','stay_id']].copy();f['event']=rebuilt['future_source_id'][mask,col]
        if f.event.eq('').any():raise ValueError('missing paired event identity')
        counts=f.groupby(['stay_id','event']).size()
        report['targets'][var]=dict(rows=len(f),patients=int(f.subject_id.nunique()),unique_future_events=len(counts),
            excess_row_fraction=float(1-len(counts)/len(f)),max_queries_per_event=int(counts.max()),
            p95_queries_per_event=float(counts.quantile(.95)))
    Path('artifacts/auxiliary_event_reuse_20260911.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

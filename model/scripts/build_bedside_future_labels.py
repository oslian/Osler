"""Measured bedside labels on the existing queries; no query selection changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=root/'bedside_future_labels.npz';output=Path('artifacts/bedside_future_labels_20260912.json')
    if path.exists() or output.exists():raise ValueError('fresh outputs required')
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    events=pd.read_parquet(root/'qualified_bedside_events.parquet')
    variables=['heart_rate','map','o2sat','respiratory_rate','temperature']
    labels=future_pairs(q,events,variables,horizon=6.,tolerance=1.,max_age=6.)
    np.savez_compressed(path,**labels,variables=np.asarray(variables))
    summary={v:dict(pairs=int(labels['paired'][:,i].sum()),patients=int(q.loc[labels['paired'][:,i],'uniquepid'].nunique())) for i,v in enumerate(variables)}
    output.write_text(json.dumps(dict(complete=True,training_performed=False,runtime_modified=False,
        queries=len(q),targets=summary,policy='6h plus/minus1h; latest known current; nearest measured future rejects quality/conflicts',
        bindings={str(p):sha256(p) for p in [path,root/'qualified_bedside_events.parquet',root/'queries.parquet',
            root/'labels.npz',Path(__file__),Path('whole_body_jepa/availability_supervision.py')]}),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

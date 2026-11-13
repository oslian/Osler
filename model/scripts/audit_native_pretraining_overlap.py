"""Check current source-file identifier overlap without publishing identifiers."""
import json
from pathlib import Path
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/native_pretraining_overlap_20260913.json')
    if output.exists():raise ValueError('fresh output required')
    checkpoint=Path('models/whole_body_joint_jepa.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    source=Path(saved['training']['event_examples'])
    queries=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    a=pd.read_parquet(source,columns=['subject_id','stay_id'])
    b=pd.read_parquet(queries,columns=['uniquepid','stay_id'])
    def ids(series):return set(series.dropna().astype(str))
    result=dict(complete=True,source_subjects=len(ids(a.subject_id)),query_subjects=len(ids(b.uniquepid)),
        exact_subject_overlap=len(ids(a.subject_id)&ids(b.uniquepid)),
        exact_stay_overlap=len(ids(a.stay_id)&ids(b.stay_id)),
        historical_source_hash_available=False,source_rows=len(a),checkpoint_training_rows=saved['training']['rows'],
        caveat='Current files only; no saved original input hash or identity-remapping audit. No untouched-pretraining claim.',
        runtime_modified=False,bindings={str(p):sha256(p) for p in (checkpoint,source,queries,Path(__file__))})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='bindings'})


if __name__=='__main__':main()

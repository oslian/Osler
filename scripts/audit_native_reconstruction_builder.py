"""Real-source smoke audit of query-native multi-target reconstruction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');cache=root/'map_native_anchor_support'
    output=Path('artifacts/native_reconstruction_builder_20260913.json')
    if output.exists():raise ValueError('fresh output required')
    q=pd.read_parquet(cache/'queries.parquet');fit=np.load(cache/'partition_7.npz')['fit']
    patients=sorted(q.loc[fit,'uniquepid'].unique())[:8]
    stays=q.loc[q.uniquepid.isin(patients),'stay_id'].unique()
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    events=pd.concat([original,bedside],ignore_index=True)
    events=events.loc[events.stay_id.isin(stays)&events.source_kind.isin(['lab_available','bedside_available'])]
    anchors=events[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];n=len(variables)
    counts={v:0 for v in variables};total=0
    for example in reconstruction_examples(events,anchors,variables,assets.scaler.medians,
            assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
        col=example['target_index'];x=example['tokens'];t=example['event_hours']
        if not np.isfinite(example['truth_normalized']) or (t>example['anchor_hour']).any():raise ValueError('illegal example')
        if (x[:,n+col]!=0).any():raise ValueError('unmasked target')
        counts[variables[col]]+=1;total+=1
    paths=[Path(__file__),Path('whole_body_jepa/native_reconstruction_examples.py'),
        Path('whole_body_jepa/native_history_schedule.py'),Path('whole_body_jepa/native_event_tokens.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),root/'native_events.parquet',root/'qualified_bedside_events.parquet',
        cache/'queries.parquet',cache/'partition_7.npz']
    report=dict(complete=True,patients=len(patients),anchors=len(anchors),examples=total,
        target_counts=counts,training_performed=False,runtime_modified=False,
        scope='first eight sorted source-fit patients; builder smoke, not efficacy',
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()

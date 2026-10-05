"""Real-source smoke audit of query-native multi-target reconstruction."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');cache=root/'map_native_anchor_support'
    output=Path('artifacts/native_reconstruction_cache_equivalence_20260913.json')
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
    source=Path('whole_body_jepa/native_reconstruction_examples.py').read_text()
    old=source.replace('n=len(variables);encoded={}','n=len(variables)')
    start=old.index('            key=tuple(group.tolist())')
    end=old.index('            tokens.append(token)',start)
    old=old[:start]+"""            part=canonical.iloc[group]
            token,_=encode_group(part,variables,medians,scales,module_ids,module_count,max_age_hours)
            hour=(pd.Timestamp(part.event_time.iloc[0])-pd.Timestamp('2100-01-01')).total_seconds()/3600
            token[3*n]=hour/24;token[-1]=hour/168
"""+old[end:]
    if hashlib.sha256(old.encode()).hexdigest()!='e0849cbe4e59f8a84be290fd99477d8db4df980df744d9c4ed06bcb14ad408e3':
        raise ValueError('historical reference not reproduced')
    ns={};exec(compile(old,'historical_reference','exec'),ns)
    def run(builder,queries):
        digest=hashlib.sha256();counts={v:0 for v in variables};start=time.perf_counter()
        for e in builder(events,queries,variables,assets.scaler.medians,assets.scaler.scales,
                         assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
            digest.update(np.asarray([e['query_position'],e['target_index'],e['truth_normalized'],e['anchor_hour']],np.float64).tobytes())
            digest.update(e['tokens'].numpy().tobytes());digest.update(e['event_hours'].numpy().tobytes())
            counts[variables[e['target_index']]]+=1
        return dict(digest=digest.hexdigest(),counts=counts,seconds=time.perf_counter()-start)
    reference=run(ns['reconstruction_examples'],anchors.iloc[:64])
    cached=run(reconstruction_examples,anchors.iloc[:64])
    if reference['digest']!=cached['digest']:raise ValueError('sample contents changed')
    full=run(reconstruction_examples,anchors)
    prior=json.loads(Path('artifacts/native_reconstruction_builder_20260913.json').read_text())
    if full['counts']!=prior['target_counts']:raise ValueError('full target counts changed')
    paths=[Path(__file__),Path('whole_body_jepa/native_reconstruction_examples.py'),
        Path('whole_body_jepa/native_history_schedule.py'),Path('whole_body_jepa/native_event_tokens.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),root/'native_events.parquet',root/'qualified_bedside_events.parquet',
        cache/'queries.parquet',cache/'partition_7.npz']
    report=dict(complete=True,patients=len(patients),anchors=len(anchors),
        reference=reference,cached=cached,full_cached=full,
        matched_anchor_count=64,full_counts_reproduced=True,
        training_performed=False,runtime_modified=False,
        bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()


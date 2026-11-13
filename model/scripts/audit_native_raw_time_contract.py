"""Count invalid original native histories before any target masking."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/native_raw_time_contract_20260913.json')
    if output.exists():
        raise ValueError('fresh output required')
    bindings={}
    def bind(path):
        bindings[str(path)]=sha256(path)
        return path
    q=pd.read_parquet(bind(root/'queries.parquet'))
    hours=[np.array([np.nan])]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            with np.load(bind(root/'unclipped_clock_shards'/p.name)) as c:
                hours.append(c['event_hours'])
            with np.load(bind(p)) as a, np.load(bind(root/'native_query_shards'/p.name)) as b:
                yield dict(a),dict(b)
    raw=RaggedNativeBatch(shards(),len(q))
    hours=np.concatenate(hours)
    anchors=((pd.to_datetime(q['t'])-pd.Timestamp('2100-01-01')).dt.total_seconds()/3600).to_numpy()
    counts=dict(empty=0,nonfinite=0,future=0,not_strictly_ordered=0)
    for seq,anchor in zip(raw.sequences,anchors):
        times=hours[seq]
        counts['empty']+=int(len(times)==0)
        counts['nonfinite']+=int(not np.isfinite(times).all() or not np.isfinite(anchor))
        counts['future']+=int(np.any(times>anchor))
        counts['not_strictly_ordered']+=int(np.any(np.diff(times)<=0))
    bind(Path(__file__))
    bind(Path('whole_body_jepa/native_body_reconstruction.py'))
    report=dict(queries=len(q),invalid_stream_counts=counts,bindings=bindings,
                accuracy_measured=False,runtime_changed=False)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':
    main()

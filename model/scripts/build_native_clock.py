"""Build private clock-corrected inputs; preserve the trained baseline cache."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_clock import restore_clock


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    dest=root/'native_clock_tokens.npz';out=Path('artifacts/native_clock_tokens_20260911.json')
    if dest.exists() or out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
    for name in ('native_events.parquet','native_schedule.npz','native_tokens.npz'):
        if sha256(root/name)!=ref['binding'][str(root/name)]:raise ValueError('cache changed')
    events=pd.read_parquet(root/'native_events.parquet')
    with np.load(root/'native_schedule.npz') as f:s={k:f[k] for k in f.files}
    with np.load(root/'native_tokens.npz') as f:t={k:f[k] for k in f.files}
    first=s['event_indices'][s['group_ptr'][:-1]]
    hours=(pd.to_datetime(events.event_time.iloc[first])-pd.Timestamp('2100-01-01')).dt.total_seconds().to_numpy()/3600
    old=t['input'];t['input']=restore_clock(old,t['history_indices'],hours,53)
    changed=np.flatnonzero(np.any(old!=t['input'],axis=0)).tolist()
    if not set(changed)<=set([159,171]):raise ValueError('non-clock change')
    np.savez_compressed(dest,**t)
    report=dict(complete=True,training_performed=False,runtime_modified=False,changed_columns=changed,
                tokens=len(old),changed_tokens=int(np.any(old!=t['input'],axis=1).sum()),
                binding={str(p):sha256(p) for p in (dest,root/'native_tokens.npz',root/'native_schedule.npz',root/'native_events.parquet',Path(__file__),Path('whole_body_jepa/native_clock.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

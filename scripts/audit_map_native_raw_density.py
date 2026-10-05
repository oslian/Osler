"""Compare legal raw draw times with fixed-lag draw-time proxies."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.audit_map_native_history_density import draw_counts
from whole_body_jepa.raw_lab_replay import canonical_events
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS
from whole_body_jepa.multisource_intervals import sha256


def raw_counts(queries, events, variables):
    queries = queries.reset_index(drop=True)
    out = np.zeros((len(queries),len(variables)),int)
    events = canonical_events(events)
    groups = dict(tuple(events.groupby('stay_id',sort=False)))
    positions = {v:i for i,v in enumerate(variables)}
    for stay,q in queries.groupby('stay_id',sort=False):
        raw = groups.get(stay)
        if raw is None:
            continue
        stream = list(raw.loc[raw.variable.isin(variables)].itertuples())
        cursor = 0
        versions, times = {}, {}
        for row,t in zip(q.sort_values('t').index,q.sort_values('t').t):
            changed = set()
            while cursor < len(stream) and stream[cursor].available_at <= t:
                e = stream[cursor]
                versions.setdefault(e.variable,{})[str(e.source_id)] = e
                changed.add(e.variable)
                cursor += 1
            for var in changed:
                by_time = {}
                for e in versions[var].values():
                    by_time.setdefault(e.measured_at,[]).append(e)
                valid_times = []
                for measured, records in by_time.items():
                    qualities = [getattr(e,'quality_valid',True) for e in records]
                    if (all(not pd.isna(v) and bool(v) for v in qualities)
                            and len({float(e.value) for e in records}) == 1):
                        valid_times.append(measured.value)
                times[var] = np.asarray(sorted(valid_times),dtype=np.int64)
            for var,draws in times.items():
                out[row,positions[var]] = (np.searchsorted(draws,t.value,side='right')
                    - np.searchsorted(draws,(t-pd.Timedelta(hours=6)).value,side='left'))
    return out


def main():
    base = Path('/private/tmp/rate_anchored_potassium_20260911')
    root = base/'map_native_anchor_support'
    output = Path('artifacts/map_native_raw_density_20260912.json')
    if output.exists():
        raise ValueError('fresh outputs required')
    parent = Path('artifacts/map_native_anchor_inputs_20260912.json')
    ref = json.loads(parent.read_text())
    q = pd.read_parquet(root/'queries.parquet')
    valid = np.load(root/'labels.npz')['paired'][:,0]
    modelpath = root/'tree_trial/full_7.joblib'
    variables = joblib.load(modelpath)['variables']
    labs = pd.read_parquet(base/'body_labs.parquet'); labs['quality_valid'] = True
    raw = pd.concat([labs,pd.read_parquet(base/'qualified_bedside_events.parquet')],ignore_index=True)
    counts = raw_counts(q,raw,variables)
    snap = np.zeros_like(counts)
    for shard in ref['shards']:
        assert sha256(Path(shard['path'])) == shard['sha256']
        a = dict(np.load(shard['path']))
        h = a['history'][a['query_rows']]
        snap[a['global_query_rows']] = draw_counts(a['mask'][h],a['ages'][h],HISTORY_LAGS_HOURS,6.)
    # Snapshots can legitimately preserve earlier versions invalidated later;
    # do not force a subset relation using future knowledge.
    private = root/'raw_density.npz'
    np.savez_compressed(private, raw_count=counts, snapshot_count=snap)
    roles = dict(np.load(root/'partition_7.npz'))
    summary = {}
    for role,mask in roles.items():
        m = mask & valid
        def average(a):
            return pd.DataFrame(a[m]).groupby(q.loc[m,'uniquepid'].to_numpy()).mean().mean().to_numpy()
        rp,sp,miss,more = [average(a) for a in (counts>=2,snap>=2,(counts>=2)&(snap<2),snap>counts)]
        summary[role] = [dict(variable=v,raw_repeat_fraction=float(rp[j]),snapshot_repeat_fraction=float(sp[j]),
            raw_repeated_but_snapshot_not=float(miss[j]),snapshot_more_than_current_raw=float(more[j])) for j,v in enumerate(variables)]
    bindings = [parent,Path(__file__),Path('scripts/audit_map_native_history_density.py'),
                base/'body_labs.parquet',base/'qualified_bedside_events.parquet',private,
                root/'queries.parquet',root/'labels.npz',root/'partition_7.npz',modelpath,
                Path('whole_body_jepa/raw_lab_replay.py')]
    output.write_text(json.dumps(dict(complete=True,roles=summary,training_performed=False,runtime_modified=False,
        caveat='Raw as-of latest versions versus historically known snapshots; descriptive seed7 only.',
        bindings={str(p):sha256(p) for p in bindings}),indent=2)+'\n')
    for role in ('fit','hospital'):
        print(role,[v for v in summary[role] if v['variable'] in ('map','glucose','heart_rate','creatinine','lactate','hemoglobin','bilirubin')])


if __name__ == '__main__':
    main()

"""Private variable-specific snapshots of the exact target-masked histories."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.extract_additional_observation_support import support
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.reconstruction_cache import ReconstructionCache, unpack_example


def snapshot(example, n):
    support(example, n)  # Same availability, mask and target visibility invariants.
    x = example['tokens'].numpy()
    times = example['event_hours'].numpy()
    mask = x[:, n:2*n].astype(bool)
    ages = example['anchor_hour'] - times[:, None] + 24*x[:, 2*n:3*n]
    if np.any(np.diff(times) < 0) or not np.isfinite(x[:, :n][mask]).all():
        raise ValueError('Invalid ordered numeric history')
    observed = mask.any(0)
    eligible = np.where(mask, ages, np.inf)
    # Most recent measurement, NOT most recently received older result.
    # Equal measurement age uses the latest available version in the cache.
    row = len(x) - 1 - np.argmin(eligible[::-1], axis=0)
    values = x[row, np.arange(n)].astype(float)
    latest_age = ages[row, np.arange(n)].astype(float)
    values[~observed] = np.nan
    latest_age[~observed] = np.nan
    return values, latest_age, observed


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    cp = root/'full_histories_20260922.sqlite'
    sp = root/'scaling_20260922.pt'
    rp = Path('artifacts/native_reconstruction_additional_observation_support_20260925.json')
    report = json.loads(rp.read_text())
    for path, digest in report['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed support dependency')
    variables = torch.load(sp, weights_only=False)['variables']
    dest = root/'variable_snapshot_20260925.parquet'
    out = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    tmp = dest.with_suffix('.tmp')
    if dest.exists() or out.exists() or tmp.exists():
        raise ValueError('Existing snapshot output; inspect before resuming')
    columns = ([f'value_{v}' for v in variables] + [f'age_{v}' for v in variables]
               + [f'observed_{v}' for v in variables])
    cache = ReconstructionCache(cp)
    writer = None
    rows = []
    count = 0
    try:
        for patient, target, payload, digest in cache.db.execute(
                'SELECT patient,target,payload,digest FROM examples ORDER BY patient,target'):
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError('Corrupt history')
            example = unpack_example(payload)
            if example['target_index'] != target:
                raise ValueError('Wrong target')
            values, ages, observed = snapshot(example, len(variables))
            rows.append((patient, variables[target], *values, *ages, *observed))
            count += 1
            if len(rows) == 10000:
                table = pa.Table.from_pandas(pd.DataFrame(rows, columns=['patient', 'target', *columns]), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
                writer.write_table(table)
                rows.clear()
            if count % 200000 == 0:
                print('Snapshot histories', count, flush=True)
        if rows:
            table = pa.Table.from_pandas(pd.DataFrame(rows, columns=['patient', 'target', *columns]), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        cache.close()
        if writer is not None:
            writer.close()
    identity = pd.read_parquet(tmp, columns=['patient', 'target'])
    expected = pd.read_parquet(root/'observation_support_20260925.parquet', columns=['patient', 'target'])
    if identity.duplicated().any() or not identity.equals(expected):
        raise ValueError('Snapshot identity mismatch')
    tmp.replace(dest)
    result = dict(pairs=count, variables=variables, private_sha256=sha256(dest),
                  bindings={str(p): sha256(p) for p in (Path(__file__), cp, sp, rp)},
                  runtime_modified=False, promotion_allowed=False,
                  semantics='Latest measured value among known events; latest available version breaks exact ties. Missing stays NaN with explicit mask. No target truth features.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(pairs=count, variables=len(variables), complete=True)), flush=True)


if __name__ == '__main__':
    main()

"""Source-only replay of the fixed patient-balanced training schedule."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_event_aware_patient_state import patient_partition
from whole_body_jepa.multisource_intervals import sha256


def replay(pools, seed, epochs=32, steps=16, batch_size=32):
    rng = np.random.default_rng(seed)
    counts = np.zeros(max(int(p.max()) for p in pools) + 1, dtype=np.int64)
    for _ in range(epochs * steps):
        selected = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=batch_size)])
        np.add.at(counts, selected, 1)
    return counts


def main():
    base_path = Path('/private/tmp/availability_glucose_development.parquet')
    extra_path = Path('/private/tmp/encoder_source_expansion_20260910/queries.parquet')
    base = pd.read_parquet(base_path)
    base = base.loc[base.fresh & base.truth.notna()].reset_index(drop=True)
    extra = pd.read_parquet(extra_path)
    extra = extra.loc[extra.fresh & extra.truth.notna()].reset_index(drop=True)
    order = sorted(set(extra.subject_id), key=lambda s: hashlib.sha256(('learning-curve-v1:' + s).encode()).hexdigest())
    report = dict(status='diagnostic_only', runtime_modified=False, external_results_used=False,
                  binding={str(p): sha256(p) for p in (base_path, extra_path, Path(__file__))}, sizes={})
    for count, name in ((0, 'history_mask'), (750, 'source1000'), (2000, 'source2250')):
        source_path = Path(f'artifacts/encoder_{name}_20260910.json')
        source = json.loads(source_path.read_text())
        if not source.get('complete'):
            raise ValueError('training incomplete')
        for path in (base_path, extra_path) if count else (base_path,):
            if source['data_binding'][str(path)] != sha256(path):
                raise ValueError('training inputs changed')
        report['binding'][str(source_path)] = sha256(source_path)
        addition = extra.loc[extra.subject_id.isin(order[:count])]
        f = pd.concat([base, addition], ignore_index=True)
        records = []
        for seed_run in source['seeds']:
            seed = seed_run['seed']
            fit_base, _ = patient_partition(base.subject_id, seed)
            fit = np.r_[fit_base, np.ones(len(addition), dtype=bool)]
            pools = [np.flatnonzero(fit & f.subject_id.eq(s).to_numpy()) for s in sorted(set(f.loc[fit, 'subject_id']))]
            hits = replay(pools, seed)
            hits = np.pad(hits, (0, len(f) - len(hits)))
            seen = hits > 0
            if np.any(seen & ~fit) or hits.sum() != 16384:
                raise ValueError('schedule mismatch')
            group = pd.Series(hits).groupby(f.subject_id).sum()
            group = group.reindex(sorted(set(f.loc[fit, 'subject_id'])))
            event_keys = ['stay_id', 'future_event_time', 'future_available_at']
            records.append(dict(seed=seed, fit_patients=len(pools), fit_rows=int(fit.sum()),
                drawn_rows=int(hits.sum()), unique_rows_seen=int(seen.sum()),
                row_coverage=float(seen.sum()/fit.sum()),
                patients_seen=int((group>0).sum()), draws_per_patient_quantiles=np.quantile(group,[0,.25,.5,.75,1]).tolist(),
                future_event_count=len(f.loc[fit,event_keys].drop_duplicates()),
                future_events_seen=len(f.loc[seen,event_keys].drop_duplicates()),
                curves={arm:dict(best_epoch=t['best_epoch'],
                    early_tune=float(np.mean([v['tune_mae'] for v in t['log'][8:16]])),
                    late_tune=float(np.mean([v['tune_mae'] for v in t['log'][24:32]])),
                    early_sample_loss=float(np.mean([v['value_loss'] for v in t['log'][8:16]])),
                    late_sample_loss=float(np.mean([v['value_loss'] for v in t['log'][24:32]])))
                    for arm,t in seed_run['training'].items()}))
        report['sizes'][str(250+count)] = records
    report['complete'] = True
    Path('artifacts/encoder_exposure_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

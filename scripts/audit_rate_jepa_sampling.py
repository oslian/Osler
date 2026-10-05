"""Replay the exact source sampler without fitting or reading test outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/rate_jepa_sampling_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    reference=Path('artifacts/rate_anchor_jepa_20260911.json')
    report=dict(complete=False,fitting_performed=False,runtime_modified=False,
                binding={str(reference):sha256(reference)},seeds=[])
    for entry in json.loads(reference.read_text())['seeds']:
        seed=entry['seed'];fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]
        pools=[np.flatnonzero(fit&q.uniquepid.eq(p).to_numpy()) for p in sorted(set(q.loc[fit,'uniquepid']))]
        rng=np.random.default_rng(seed)
        draws=np.concatenate([np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)]) for _ in range(32*16)])
        seen=np.unique(draws);ids=labels['future_source_id'][valid,0]
        report['seeds'].append(dict(seed=seed,fit_patients=len(pools),fit_rows=int(fit.sum()),
            seen_patients=int(q.loc[seen,'uniquepid'].nunique()),row_coverage=float(len(seen)/fit.sum()),
            future_event_coverage=float(len(set(ids[seen]))/len(set(ids[fit]))),
            selected_epoch=entry['arms']['whole']['selected_epoch'],
            last_eight_tune_mae=[r['tune_mae'] for r in entry['arms']['whole']['log'][-8:]]))
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

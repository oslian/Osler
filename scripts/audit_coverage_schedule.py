"""Outcome-free exposure comparison with the original training schedule."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.bounded_site_sampling import bounded_weights, sample_indices
from scripts.coverage_sampling import coverage_indices


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    scaling = torch.load(root/'scaling_20260922.pt', weights_only=False)
    plan = torch.load(root/'partitions_20260922.pt', weights_only=False)
    provenance = pd.read_parquet(root/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if provenance.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    component = dict(provenance.itertuples(index=False, name=None))
    people, centers = scaling['people'], scaling['centers']
    results = []
    for fold in (0, 1, 3, 4):
        fit = np.array([plan['folds'][fold][p] == 'fit' for p in people])
        pools = {int(c): np.flatnonzero(fit & np.isfinite(centers[:, c]))
                 for c in np.flatnonzero(scaling['scalers'][fold]['counts'])}
        weights = {c: bounded_weights([component[people[i]] for i in ids]) for c, ids in pools.items()}
        arms = {}
        for arm in ('replacement', 'coverage'):
            rng = np.random.default_rng(19)
            seen, pending = set(), {c: [] for c in pools}
            for epoch in range(8):
                schedule = []
                for c, ids in pools.items():
                    if arm == 'coverage':
                        draws, pending[c] = coverage_indices(weights[c], rng, 1072, pending[c])
                    else:
                        draws = sample_indices(weights[c], rng, 1072)
                    schedule.extend((int(ids[i]), c) for i in draws)
                rng.shuffle(schedule)
                seen.update(schedule)
            if arm == 'replacement':
                checkpoint = torch.load(root/f'training_20260923/fold{fold}_real.pt', weights_only=False)
                if seen != checkpoint['seen']:
                    raise ValueError('Original schedule does not reproduce checkpoint')
            arms[arm] = dict(unique_pairs=len(seen),
                             per_target={scaling['variables'][c]: sum(j == c for _, j in seen) for c in pools})
        results.append(dict(fold=fold, eligible_pairs=sum(map(len, pools.values())), arms=arms))
    output = Path('artifacts/native_reconstruction_coverage_schedule_20260925.json')
    if output.exists():
        raise ValueError('Existing schedule audit')
    output.write_text(json.dumps(dict(results=results, training_performed=False,
        limitation='Sampling-only simulation. Weights affect ordering, not complete-cycle frequency. No accuracy evidence or promotion.'), indent=2)+'\n')
    print(json.dumps([dict(fold=r['fold'], eligible=r['eligible_pairs'],
                          **{a: x['unique_pairs'] for a, x in r['arms'].items()}) for r in results], indent=2))


if __name__ == '__main__':
    main()

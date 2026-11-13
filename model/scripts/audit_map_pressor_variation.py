"""Source-fit recorded rate support and variation, without outcome inspection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/map_recorded_pressors_20260912.json')
    for p, digest in json.loads(parent.read_text())['bindings'].items():
        assert sha256(Path(p)) == digest
    data = np.load(root / 'map_recorded_pressors.npz')
    q = pd.read_parquet(root / 'queries.parquet')
    eligible = np.load(root / 'map_all_query_labels.npz')['paired'][:, 0]
    reports = []; paths = [Path(__file__), parent]
    for seed in (7, 19, 31, 43, 59, 71, 83):
        partition = root / f'partition_{seed}.npz'; paths.append(partition)
        fit = np.load(partition)['fit'] & eligible
        columns = {}
        for j, drug in enumerate(data['drugs']):
            for k, dimension in enumerate(data['dimensions']):
                rate = data['rate'][:, j, k]; mask = fit & np.isfinite(rate)
                values = rate[mask]
                if not len(values): continue
                columns[str(drug) + ':' + str(dimension)] = dict(
                    queries=int(mask.sum()), patients=int(q.loc[mask, 'uniquepid'].nunique()),
                    distinct_recorded_rates=int(len(np.unique(values))),
                    zero_fraction=float(np.mean(values == 0)),
                    quartiles=np.quantile(values, [.25, .5, .75]).tolist())
        reports.append(dict(seed=seed, columns=columns))
    output = Path('artifacts/map_pressor_variation_20260912.json')
    output.write_text(json.dumps(dict(results=reports, role='source_fit_only', training_performed=False,
        runtime_modified=False, online_eligible=False,
        limitations=['Repeated carried rates do not establish independent administration events.',
                     'Numerical variation is not predictive information or physiological validity.'],
        bindings={str(p):sha256(p) for p in paths}), indent=2) + '\n')
    names = sorted(set().union(*(r['columns'] for r in reports)))
    for name in names:
        cs = [r['columns'][name] for r in reports if name in r['columns']]
        print(name, 'splits',len(cs),'patients',min(c['patients'] for c in cs),max(c['patients'] for c in cs),
              'unique_rates',min(c['distinct_recorded_rates'] for c in cs),max(c['distinct_recorded_rates'] for c in cs))


if __name__ == '__main__': main()

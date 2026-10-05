"""Outcome-free documented pressor support in the expanded MAP cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.multisource_intervals import sha256

TERMS = {
    'norepinephrine': r'norepinephrine|noradrenaline|levophed',
    'epinephrine': r'(?<!nor)\bepinephrine\b|(?<!nor)\badrenaline\b',
    'phenylephrine': r'phenylephrine|neosynephrine|neo-synephrine',
    'dopamine': r'\bdopamine\b',
    'vasopressin': r'vasopressin',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/map_pressor_inventory_20260912.json'))
    output = parser.parse_args().output
    if output.exists(): raise ValueError('fresh output required')
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    raw = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    q = pd.read_parquet(root / 'queries.parquet')
    eligible = np.load(root / 'map_all_query_labels.npz')['paired'][:, 0]
    q = q.loc[eligible].copy()
    q['raw_stay'] = q.stay_id.str.removeprefix('eicu:').astype(int)
    stays = set(q.raw_stay)
    records = []; matched = {k: 0 for k in TERMS}; columns = None
    for chunk in pd.read_csv(raw, chunksize=500000, low_memory=False):
        columns = list(chunk.columns)
        chunk = chunk[chunk.patientunitstayid.isin(stays)]
        names = chunk.drugname.fillna('').str.lower()
        for drug, pattern in TERMS.items():
            part = chunk[names.str.contains(pattern, regex=True)]
            matched[drug] += len(part)
            for row in part.itertuples(index=False):
                parsed = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate',
                    rate_like=row.drugrate, original_label=row.drugname), 'vasopressor')
                rate = parsed['action_rate']; offset = pd.to_numeric(row.infusionoffset, errors='coerce')
                if np.isfinite(rate) and np.isfinite(offset):
                    records.append((int(row.patientunitstayid), row.infusiondrugid, drug,
                        parsed['action_rate_dimension'], float(offset), float(rate)))
    events = pd.DataFrame(records, columns=['stay', 'id', 'drug', 'dimension', 'offset', 'rate'])
    events = events.drop_duplicates(['id', 'drug'])
    inventory = {}; any_support = np.zeros(len(q), bool)
    for drug in TERMS:
        part = events[events.drug.eq(drug)]
        support = np.zeros(len(q), bool)
        groups = {s: np.sort(g.offset.to_numpy()) for s, g in part.groupby('stay')}
        for i, row in enumerate(q.itertuples(index=False)):
            times = groups.get(row.raw_stay)
            if times is None: continue
            pos = np.searchsorted(times, row.offset, side='right') - 1
            support[i] = pos >= 0 and row.offset - times[pos] <= 360
        any_support |= support
        inventory[drug] = dict(raw_matching_rows=matched[drug], compatible_rate_rows=len(part),
            dimensions={str(k): int(v) for k, v in part.dimension.value_counts().items()},
            preanchor6h_queries=int(support.sum()),
            preanchor6h_patients=int(q.loc[support, 'uniquepid'].nunique()))
    output.write_text(json.dumps(dict(status='inventory_only', source_columns=columns,
        map_queries=len(q), map_patients=int(q.uniquepid.nunique()), drugs=inventory,
        any_preanchor6h_queries=int(any_support.sum()),
        any_preanchor6h_patients=int(q.loc[any_support, 'uniquepid'].nunique()),
        independent_availability_verified=False, runtime_modified=False,
        limitations=['Recorded infusionoffset is not independently verified result availability.',
                     'Support is not confirmed active infusion or delivered dose.',
                     'No future outcomes used for rate inventory; cohort uses MAP eligibility.',
                     'Drug families and rate dimensions remain separate; no equipotent sum.'],
        bindings={str(p): sha256(p) for p in [Path(__file__), raw, root/'queries.parquet',
            root/'map_all_query_labels.npz', Path('whole_body_jepa/event_treatment.py'),
            Path('osler_jepa/treatment_units.py')]}), indent=2) + '\n')
    print(output.read_text())


if __name__ == '__main__': main()

"""Source-only site support and fixed-epoch weighting sensitivity."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_continued_site_transfer import site_summary
from whole_body_jepa.multisource_intervals import sha256


def unique_site_map(frame):
    required = ['uniquepid', 'hospitalid', 'role']
    if frame[required].isna().any().any():
        raise ValueError('Missing provenance')
    if frame.groupby('uniquepid').role.nunique().max() != 1:
        raise ValueError('Patient crosses source roles')
    sites = frame.groupby('uniquepid').hospitalid.agg(lambda x: set(map(str, x)))
    return {p: next(iter(s)) for p, s in sites.items() if len(s) == 1}, {p for p, s in sites.items() if len(s) != 1}


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_source_site_alignment_20260920.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    reports = [Path('artifacts/native_reconstruction_full_broad_samples_20260918.json'),
               Path('artifacts/native_reconstruction_expanded_selection_20260919.json'),
               Path('artifacts/native_reconstruction_continued_selection_20260920.json')]
    pp = root / 'provenance.parquet'
    paths = [root / 'expanded_selection_replay/epoch7.pt', root / 'continued_selection_replay/epoch16.pt']
    first, old, new = [json.loads(p.read_text()) for p in reports]
    if sha256(pp) != first['bindings'][str(pp)]:
        raise ValueError('Provenance changed')
    for p, report in zip(paths, (old, new)):
        if sha256(p) != report['private_predictions'][str(p)]:
            raise ValueError('Replay changed')
    provenance = pd.read_parquet(pp)
    mapping, ambiguous = unique_site_map(provenance)
    role_counts = {}
    for role, frame in provenance.groupby('role'):
        people = set(frame.uniquepid)
        counts = Counter(mapping[p] for p in people if p in mapping)
        weights = np.array(list(counts.values()), dtype=float)
        weights /= weights.sum()
        role_counts[role] = dict(patients=len(people), ambiguous_patients=len(people & ambiguous),
                                sites=len(counts), largest_site_fraction=float(weights.max()),
                                effective_sites=float(1 / np.square(weights).sum()))
    records = [torch.load(p, weights_only=False)['rows'] for p in paths]
    indexed = [{(r['patient'], r['target']): r for r in rows} for rows in records]
    if any(len(d) != len(r) for d, r in zip(indexed, records)) or indexed[0].keys() != indexed[1].keys():
        raise ValueError('Pair identity mismatch')
    tune = set(provenance.loc[provenance.role.eq('encoder_tune'), 'uniquepid'])
    per = {}
    for target in old['trained_targets']:
        delta, sites, excluded = [], [], 0
        for (patient, t), a in indexed[0].items():
            if t != target:
                continue
            b = indexed[1][patient, t]
            if patient not in tune or a['truth'] != b['truth']:
                raise ValueError('Source truth/role mismatch')
            if patient in ambiguous:
                excluded += 1
                continue
            delta.append(abs(b['prediction'] - b['truth']) - abs(a['prediction'] - a['truth']))
            sites.append(mapping[patient])
        per[target] = dict(**site_summary(delta, sites), patients=len(delta), excluded_ambiguous_pairs=excluded)
    payload = dict(roles=role_counts, targets=per,
        weighting_sign_changes=sum(r['weighting_sign_disagreement'] for r in per.values()),
        runtime_modified=False, promotion_allowed=False,
        limitation='Source-only descriptive epoch7/16 comparison; excludes ambiguous hospital attribution. Not a complete-site weighting intervention or proof of causality.',
        bindings={str(p): sha256(p) for p in [*reports, pp, *paths, Path(__file__)]})
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(roles=role_counts, weighting_sign_changes=payload['weighting_sign_changes'],
                         reversed_targets=[t for t, r in per.items() if r['weighting_sign_disagreement']]), indent=2))


if __name__ == '__main__':
    main()

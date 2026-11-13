"""Locate ranking loss before external transfer using immutable source replay."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def main():
    out = Path('artifacts/native_reconstruction_source_ranking_20260917.json')
    if out.exists():
        raise ValueError('existing diagnostic')
    query = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    q = pd.read_parquet(query)
    if q.hospitalid.isna().any():
        raise ValueError('missing provenance')
    counts = q.groupby('uniquepid').hospitalid.nunique()
    known = q[q.uniquepid.isin(counts[counts == 1].index)].groupby('uniquepid').hospitalid.first()
    bindings = {str(p): sha256(p) for p in (Path(__file__), query)}
    results = {}
    for seed in (19, 31, 43, 59, 71, 83):
        path = Path(f'artifacts/native_reconstruction_source_fit_{seed}_20260917.json')
        report = json.loads(path.read_text())
        expected_query = [h for p, h in report['bindings'].items() if Path(p).name == 'queries.parquet']
        if expected_query != [sha256(query)]:
            raise ValueError('wrong cohort provenance')
        matches = [(durable(p), h) for p, h in report['bindings'].items() if Path(p).name == 'source_predictions.pt']
        if len(matches) != 1 or sha256(matches[0][0]) != matches[0][1]:
            raise ValueError('changed source predictions')
        saved = matches[0][0]
        rows = torch.load(saved, weights_only=False)
        bindings.update({str(p): sha256(p) for p in (path, saved)})
        roles = {}
        for role in ('fit', 'tune'):
            frame = pd.DataFrame(rows[role])
            if frame.duplicated(['subject_id', 'target']).any():
                raise ValueError('duplicate identity')
            frame['site'] = frame.subject_id.map(known)
            targets = {}
            for target, part in frame.groupby('target'):
                for arm in ('real', 'process', 'population'):
                    actual = float(abs(part[arm]-part.truth).mean())
                    if not np.isclose(actual, report['roles'][role][target]['normalized_mae'][arm], atol=1e-10, rtol=0):
                        raise ValueError('original score mismatch')
                sites = []
                for _, group in part.groupby('site'):
                    if len(group) < 30:
                        continue
                    sites.append(dict(patients=len(group),
                        real_spearman=correlation(group.real, group.truth),
                        process_spearman=correlation(group.process, group.truth)))
                targets[target] = dict(patients=len(part),
                    real_spearman=correlation(part.real, part.truth),
                    process_spearman=correlation(part.process, part.truth),
                    unknown_site_patients=int(part.site.isna().sum()),
                    supported_sites=sites)
            roles[role] = targets
        results[str(seed)] = roles
    out.write_text(json.dumps(dict(complete=True, seeds=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Six overlapping earlier V2 fits, NOT the new strict-site model. '
        'Selected source tune is not independent evaluation. Site support30 fixed; '
        'multi-site patients retained pooled but not arbitrarily assigned. Correlations '
        'descriptive, no significance, predictor sign reversal or new selection.'), indent=2, allow_nan=False)+'\n')
    for target in ('lipase', 'triglycerides', 'bilirubin', 'glucose', 'creatinine'):
        print(target, [(s, round(r['fit'][target]['real_spearman'], 3),
                        round(r['tune'][target]['real_spearman'], 3),
                        [round(x['real_spearman'], 3) for x in r['tune'][target]['supported_sites']])
                       for s, r in results.items()])


if __name__ == '__main__':
    main()

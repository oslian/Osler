"""Pre-anchor support and exact-site localization of fixed BUN history errors."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target


def support(tokens, times, anchor, n):
    if not len(tokens) or times.shape != (len(tokens),):
        raise ValueError('Nonempty aligned history required')
    mask = tokens[:, n:2*n].bool()
    latest_ages = []
    for col in range(n):
        observed = mask[:, col].nonzero().flatten()
        if len(observed):
            i = int(observed[-1])
            age = float(anchor - times[i]) + 24 * float(tokens[i, 2*n+col])
            if not np.isfinite(age) or age < 0:
                raise ValueError('Invalid measurement age')
            latest_ages.append(age)
    if not latest_ages:
        raise ValueError('No independent variable coverage')
    span = float(times[-1] - times[0])
    if span < 0:
        raise ValueError('Reversed history')
    return dict(events=len(tokens), span_hours=span, independent_variables=len(latest_ages),
                maximum_latest_age_hours=max(latest_ages))


def bins(meta):
    return dict(events='1-8' if meta['events'] <= 8 else '9-32' if meta['events'] <= 32 else '33+',
                span='0-6h' if meta['span_hours'] <= 6 else '6-24h' if meta['span_hours'] <= 24 else '24h+',
                latest_age='0-6h' if meta['maximum_latest_age_hours'] <= 6 else '6h+')


def stats(rows):
    return dict(patients=len(rows),
                change=interval([r['delta'] for r in rows]) if len(rows) >= 30 else None,
                mean_change=float(np.mean([r['delta'] for r in rows])),
                status='descriptive_only' if len(rows) >= 30 else 'insufficient_support')


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_bun_full_history_20260920.json')
    data = root / 'bun_full_history_20260920.pt'
    mp = root / 'site_balance_manifest_20260920.pt'
    mr = Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    jp = root / 'joint_source_manifest.pt'
    out = Path('artifacts/native_reconstruction_bun_history_strata_20260920.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    report, manifest_report = [json.loads(p.read_text()) for p in (rp, mr)]
    if sha256(data) != report['private_sha256'] or sha256(mp) != manifest_report['private_sha256']:
        raise ValueError('Changed private data')
    for p, expected in report['bindings'].items():
        if sha256(Path(p)) != expected:
            raise ValueError('Changed dependency')
    manifest = torch.load(mp, weights_only=False)
    raw = torch.load(data, weights_only=False)
    joint = torch.load(jp, weights_only=False)
    variables = joint['variables']
    col = variables.index('bun')
    output = {}
    for group, saved in raw.items():
        store = SourceStore([s for s in manifest['shards'] if s['group'] == group and s['role'] == 'encoder_tune'], [col])
        if {r['patient'] for r in saved} != {key[0] for key in store.labels}:
            raise ValueError('Patient set mismatch')
        rows = []
        for r in saved:
            e = store[r['patient'], col]
            tokens, times = mask_native_target(e['tokens'], e['event_hours'], variables, 'bun')
            meta = support(tokens, times, e['anchor_hour'], len(variables))
            if meta['events'] != r['events']['full']:
                raise ValueError('History mismatch')
            site = str(manifest['sample_sites'][group, 'encoder_tune', r['patient'], col])
            rows.append(dict(site=site, delta=r['errors']['latest_each']-r['errors']['full'], **bins(meta)))
        expected = report['groups'][group]['minus_full']['latest_each']['mean']
        if not np.isclose(np.mean([r['delta'] for r in rows]), expected, atol=1e-10, rtol=0):
            raise ValueError('Mean mismatch')
        dimensions = {}
        for name in ('site', 'events', 'span', 'latest_age'):
            dimensions[name] = {value: stats([r for r in rows if r[name] == value]) for value in sorted({r[name] for r in rows})}
        site_means = [s['mean_change'] for s in dimensions['site'].values()]
        dimensions['equal_site_mean'] = float(np.mean(site_means))
        dimensions['site_event_cells'] = {site: {b: stats([r for r in rows if r['site'] == site and r['events'] == b])
            for b in sorted({r['events'] for r in rows if r['site'] == site})} for site in dimensions['site']}
        output[group] = dimensions
    out.write_text(json.dumps(dict(groups=output, bindings={str(p): sha256(p) for p in (Path(__file__), rp, data, mp, mr, jp)},
        runtime_modified=False, promotion_allowed=False,
        limitation='Overlapping source-only descriptive strata with unadjusted paired intervals. Fixed pre-anchor bins; not fitted routes, clinical causality or external validation.'), indent=2) + '\n')
    for g, r in output.items():
        print(g, {k: r[k] for k in ('site', 'events', 'span', 'latest_age', 'equal_site_mean')})


if __name__ == '__main__':
    main()

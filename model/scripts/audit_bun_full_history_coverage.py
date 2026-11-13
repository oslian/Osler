"""Replicate BUN history coverage intervention on all source selection patients."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_continued_selection_pairs import interval
from scripts.audit_reconstruction_latest_coverage import latest_coverage
from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def summarize(rows):
    if not rows or len({r['patient'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty patient pairs required')
    names = ('full', 'latest_each', 'last8_latest_each')
    return dict(patients=len(rows),
        mae={n: float(np.mean([r['errors'][n] for r in rows])) for n in names},
        minus_full={n: interval([r['errors'][n] - r['errors']['full'] for r in rows]) for n in names[1:]},
        mean_events={n: float(np.mean([r['events'][n] for r in rows])) for n in names})


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    tp = Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    pp = Path('artifacts/native_reconstruction_latest_coverage_20260920.json')
    cp = root / 'bounded_site_training/bounded.pt'
    out = Path('artifacts/native_reconstruction_bun_full_history_20260920.json')
    private = root / 'bun_full_history_20260920.pt'
    prior_private = root / 'latest_coverage_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing audit')
    training, prior = [json.loads(p.read_text()) for p in (tp, pp)]
    for report in (training, prior):
        for p, h in report['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError('Changed dependency')
    if sha256(prior_private) != prior['private_sha256']:
        raise ValueError('Prior rows changed')
    previous = torch.load(prior_private, weights_only=False)
    ck = torch.load(cp, weights_only=False)
    if ck['bindings'] != training['bindings']:
        raise ValueError('Checkpoint mismatch')
    paths = (root / 'joint_source_manifest.pt', root / 'full_samples/index.pt',
             Path('data/research_recovery/reconstruction_nested_selection_20260917.pt'))
    joint, full, split = [torch.load(p, weights_only=False) for p in paths]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    original = {s['patient'] for s in shards if s['group'] == 'broad' and s['role'] == 'encoder_tune'}
    selected = [s for s in shards if s['group'] == 'selected' and s['role'] == 'encoder_tune']
    broad = [s for s in full['shards'] if s['role'] == 'encoder_tune']
    forbidden = {s['patient'] for s in shards if s['role'] == 'fit'}
    identities = [s['patient'] for s in selected + broad]
    if len(identities) != len(set(identities)) or set(identities) & forbidden:
        raise ValueError('Source overlap')
    variables = joint['variables']
    col = variables.index('bun')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw = {}
    with torch.inference_mode():
        for group, source in (('selected', selected), ('broad', broad)):
            store = SourceStore(source, [col])
            rows = []
            old = {r['patient']: r for r in previous[group] if r['target'] == 'bun'}
            for key, truth in store.labels.items():
                e = store[key]
                tokens, times = mask_native_target(e['tokens'], e['event_hours'], variables, 'bun')
                variants = {'full': (tokens, times),
                            'latest_each': latest_coverage(tokens, times, len(variables), 1),
                            'last8_latest_each': latest_coverage(tokens, times, len(variables), 8)}
                errors = {n: abs(float(model(x, t, e['anchor_hour'], 'bun')[0]) - truth) for n, (x, t) in variants.items()}
                events = {n: len(x) for n, (x, _) in variants.items()}
                if key[0] in old:
                    for n in variants:
                        if not np.isclose(errors[n], old[key[0]]['errors'][n], atol=1e-7, rtol=0) or events[n] != old[key[0]]['events'][n]:
                            raise ValueError('Original patient replay mismatch')
                rows.append(dict(patient=key[0], errors=errors, events=events, original=key[0] in original))
            if not set(old).issubset({r['patient'] for r in rows}):
                raise ValueError('Original patients missing')
            raw[group] = rows
            print(group, len(rows), 'complete', flush=True)
    result = {g: summarize(rows) for g, rows in raw.items()}
    result['broad_original'] = summarize([r for r in raw['broad'] if r['original']])
    result['broad_additional'] = summarize([r for r in raw['broad'] if not r['original']])
    torch.save(raw, private)
    out.write_text(json.dumps(dict(groups=result,
        bindings={str(p): sha256(p) for p in (Path(__file__), Path('scripts/audit_reconstruction_latest_coverage.py'), tp, pp, cp, prior_private, *paths)},
        private_sha256=sha256(private), runtime_modified=False, promotion_allowed=False,
        limitation='Source-selection replication, fixed previously selected checkpoint. Descriptive unadjusted paired bootstrap, not an untouched external test. '
        'Event deletion changes initialization, co-measurement history and input distribution, not isolated causal memory.'), indent=2) + '\n')
    print({g: r['minus_full'] for g, r in result.items()})


if __name__ == '__main__':
    main()

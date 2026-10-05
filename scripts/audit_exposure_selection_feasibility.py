"""Source-only common checkpoint feasibility; never deploy per-target oracles."""
import json
from pathlib import Path
import numpy as np
from scripts.audit_joint_retention_feasibility import classify
from whole_body_jepa.multisource_intervals import sha256


def trajectory(arm):
    groups = list(arm['initial_scores'])
    targets = list(arm['initial_scores'][groups[0]])
    epochs = [arm['initial_scores']] + [e['group_target_scores'] for e in arm['log']]
    values = np.array([[epoch[g][t] for g in groups for t in targets] for epoch in epochs])
    if not np.isfinite(values).all():
        raise ValueError('Invalid trajectory')
    return groups, targets, values


def main():
    src = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    out = Path('artifacts/native_reconstruction_exposure_selection_feasibility_20260919.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    report = json.loads(src.read_text())
    results = {}
    for mode in ('small', 'expanded'):
        arm = report['modes'][mode]['real']
        groups, targets, values = trajectory(arm)
        per = {g: {} for g in groups}
        for gi, g in enumerate(groups):
            for ti, t in enumerate(targets):
                v = values[:, gi * len(targets) + ti]
                per[g][t] = dict(category=classify(v.tolist(), arm['selected_epoch']),
                    initial=float(v[0]), selected=float(v[arm['selected_epoch']]),
                    best_observed=float(v.min()))
        results[mode] = dict(selected_epoch=arm['selected_epoch'], targets=per,
            common_epochs_nonharming_initial=np.where((values <= values[0]).all(axis=1))[0].tolist(),
            common_epochs_nonharming_selected=np.where((values <= values[arm['selected_epoch']]).all(axis=1))[0].tolist(),
            epoch_harm_counts=(values > values[0]).sum(axis=1).tolist())
    out.write_text(json.dumps(dict(modes=results, bindings={str(p): sha256(p) for p in
        (src, Path(__file__), Path('scripts/audit_joint_retention_feasibility.py'))},
        runtime_modified=False, limitation='Descriptive necessary condition on observed epochs, not a new noninferiority gate or impossibility proof.'), indent=2) + '\n')
    for mode, r in results.items():
        print(mode, 'nonharm initial', r['common_epochs_nonharming_initial'],
              'nonharm selected', r['common_epochs_nonharming_selected'], 'harms', r['epoch_harm_counts'])
        for g, ts in r['targets'].items():
            print(g, {category: [t for t, d in ts.items() if d['category'] == category]
                for category in sorted({d['category'] for d in ts.values()})})


if __name__ == '__main__':
    main()

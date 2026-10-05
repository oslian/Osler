"""Source-only selection-score attribution, never a new deployment selector."""
import json
from pathlib import Path

import numpy as np

from scripts.audit_exposure_selection_feasibility import trajectory
from whole_body_jepa.multisource_intervals import sha256


def concentration(arm):
    groups, targets, values = trajectory(arm)
    if len(targets) < 2 or (values <= 0).any():
        raise ValueError('Positive errors and at least two targets required')
    selected = arm['selected_epoch']
    if not 0 <= selected < len(values):
        raise ValueError('Invalid selected epoch')
    macro = values.mean(axis=1)
    if selected != int(np.argmin(macro)):
        raise ValueError('Recorded selection does not match source macro')
    for i, row in enumerate(arm['log'], start=1):
        if row['epoch'] != i or not np.isclose(row['tune_mae'], macro[i], atol=1e-10, rtol=0):
            raise ValueError('Recorded source score changed')
    tensor = values.reshape(len(values), len(groups), len(targets))
    score_by_target = tensor[selected].sum(axis=0) / values.shape[1]
    gain_by_target = (tensor[0] - tensor[selected]).sum(axis=0) / values.shape[1]
    per = {}
    for ti, t in enumerate(targets):
        keep = np.arange(len(targets)) != ti
        without = tensor[:, :, keep].mean(axis=(1, 2))
        per[t] = dict(selected_score_contribution=float(score_by_target[ti]),
                      selected_score_share=float(score_by_target[ti] / macro[selected]),
                      initial_to_selected_gain_contribution=float(gain_by_target[ti]),
                      leave_one_target_out_epoch=int(np.argmin(without)))
    # Reference-relative errors are diagnostic only; do not reselect deployed weights.
    relative = (values / values[0]).mean(axis=1)
    return dict(selected_epoch=selected, macro_by_epoch=macro.tolist(),
                reference_relative_macro_by_epoch=relative.tolist(),
                reference_relative_best_epoch=int(np.argmin(relative)),
                per_target=per,
                top_five_selected_score_share=float(np.sort(score_by_target)[-5:].sum() / macro[selected]),
                selected_harms_initial_count=int((values[selected] > values[0]).sum()),
                leave_one_target_out_changed=sum(x['leave_one_target_out_epoch'] != selected for x in per.values()))


def main():
    root = Path('artifacts')
    paths = [root / f'native_reconstruction_{s}_20260919.json' for s in
             ('accumulated_source', 'decay_matched_source', 'moment_clock_source')]
    pair = root / 'native_reconstruction_source_exposure_pair_20260918.json'
    reports = [json.loads(p.read_text()) for p in paths]
    paired = json.loads(pair.read_text())
    arms = {f'{mode}_sequential': x['real'] for mode, x in paired['modes'].items()}
    for prefix, r in zip(('accumulated', 'decay', 'moment'), reports):
        arms.update({f'{prefix}_{k}': x for k, x in r['arms'].items()})
    results = {k: concentration(x) for k, x in arms.items()}
    out = root / 'native_reconstruction_selection_concentration_20260919.json'
    if out.exists():
        raise ValueError('Existing audit')
    out.write_text(json.dumps(dict(arms=results, runtime_modified=False,
        limitation='Source-only descriptive sensitivity. No evaluation labels, new fitting, new selector, promotion, or gradient dominance claim.',
        bindings={str(p): sha256(p) for p in [pair, *paths, Path(__file__),
            Path('scripts/audit_exposure_selection_feasibility.py')]}), indent=2) + '\n')
    for k, r in results.items():
        top = sorted(r['per_target'], key=lambda t: r['per_target'][t]['selected_score_share'], reverse=True)[:5]
        print(k, 'selected', r['selected_epoch'], 'top5share', r['top_five_selected_score_share'],
              'targets', top, 'leaveout changes', r['leave_one_target_out_changed'],
              'relative diagnostic epoch', r['reference_relative_best_epoch'])


if __name__ == '__main__':
    main()

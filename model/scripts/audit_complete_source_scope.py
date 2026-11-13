"""Reconcile available source support with actual checkpoint training exposure."""
import json
from pathlib import Path
import torch
from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.multisource_intervals import sha256


def exposure_row(available, selected_seen, final_seen, trained):
    if not 0 <= selected_seen <= final_seen <= available:
        raise ValueError('Impossible exposure counts')
    if not trained and (selected_seen or final_seen):
        raise ValueError('Untrained head has exposure')
    return dict(available_fit=available, selected_checkpoint_seen=selected_seen,
                final_checkpoint_seen=final_seen, remaining_unseen=available-final_seen,
                final_coverage=final_seen/available if available else None,
                status='trained' if trained else 'not_trained_in_matched_trial')


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    paths = [root/'joint_source_manifest.pt', root/'full_samples/index.pt',
             Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    sample_report = Path('artifacts/native_reconstruction_full_broad_samples_20260918.json')
    training_path = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    out = Path('artifacts/native_reconstruction_complete_source_scope_20260919.json')
    if out.exists():
        raise ValueError('Existing audit')
    joint, full, split = [torch.load(p, weights_only=False) for p in paths]
    samples, training = [json.loads(p.read_text()) for p in (sample_report, training_path)]
    for report in (samples, training):
        for p, h in report['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError(f'Changed source: {p}')
    targets = list(samples['accepted']['fit'])
    variables = joint['variables']; columns = [variables.index(t) for t in targets]
    existing = source_shards(joint, full, 'expanded', split['roles'])
    counts = {}
    for g in ('selected', 'broad'):
        counts[g] = {}
        for role in ('fit', 'encoder_tune'):
            store = SourceStore([s for s in existing if s['group']==g and s['role']==role], columns)
            counts[g][role] = {variables[c]: len(keys) for c, keys in store.groups.items()}
    # New selection patients must retain their existing source-only role and not
    # overlap either source-fit group; no reassignment of evaluation patients.
    fit_ids = {s['patient'] for s in existing if s['role']=='fit'}
    selected_ids = {s['patient'] for s in existing if s['group']=='selected'}
    expanded_tune = [s for s in full['shards'] if s['role']=='encoder_tune']
    ids = [s['patient'] for s in expanded_tune]
    if len(ids)!=len(set(ids)) or set(ids) & (fit_ids | selected_ids):
        raise ValueError('Expanded selection role overlap')
    store = SourceStore(expanded_tune, columns)
    expanded_counts = {variables[c]: len(keys) for c, keys in store.groups.items()}
    if expanded_counts != samples['accepted']['encoder_tune'] or counts['broad']['fit'] != samples['accepted']['fit']:
        raise ValueError('Raw source support does not reproduce report')
    arm = training['modes']['expanded']['real']
    logs = {row['epoch']: row for row in arm['log']}
    selected = logs[arm['selected_epoch']]['unique_exposure']['broad']
    final = logs[arm['epoch']]['unique_exposure']['broad']
    rows = {}
    for t in targets:
        trained = t in training['source_targets']
        row = exposure_row(counts['broad']['fit'][t], selected.get(t, 0), final.get(t, 0), trained)
        row.update(old_selected_fit=counts['selected']['fit'][t],
                   old_selected_tune=counts['selected']['encoder_tune'][t],
                   old_broad_tune=counts['broad']['encoder_tune'][t],
                   expanded_broad_tune=expanded_counts[t])
        rows[t] = row
    out.write_text(json.dumps(dict(targets=rows, selected_epoch=arm['selected_epoch'],
        final_epoch=arm['epoch'], expanded_tune_patients=len(ids),
        bindings={str(p): sha256(p) for p in [*paths, sample_report, training_path, Path(__file__),
                                          Path('scripts/reconstruction_source_store.py')]},
        runtime_modified=False, limitation='Source support and exposure only, not training success, per-site power or new external validation.'), indent=2)+'\n')
    for t, r in rows.items():
        print(t, r, flush=True)


if __name__ == '__main__':
    main()

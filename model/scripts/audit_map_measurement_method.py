"""Retrospective MAP method-switch attribution; not deployable routing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    out = Path('artifacts/map_measurement_method_20260912.json')
    if out.exists():
        raise ValueError('fresh output required')
    parent = Path('artifacts/native_map_common_bootstrap_20260912.json')
    training = Path('artifacts/native_bedside_map_training_20260911.json')
    for source in (parent, training):
        for p, digest in json.loads(source.read_text())['bindings'].items():
            if sha256(Path(p)) != digest:
                raise ValueError('changed dependency')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'})
    labels = dict(np.load(root / 'bedside_future_labels.npz'))
    col = labels['variables'].tolist().index('map')
    q['truth'] = labels['future'][:, col]
    events_path = root / 'qualified_bedside_events.parquet'
    events = pd.read_parquet(events_path)
    events = events.loc[events.variable.eq('map')]
    groups = dict(tuple(events.groupby('stay_id')))
    strata = np.full(len(q), 'unpaired', dtype=object)
    for i in np.flatnonzero(labels['paired'][:, col]):
        row = q.iloc[i]
        ev = groups[row.stay_id]
        current_time = row.t - pd.Timedelta(hours=float(labels['measurement_age'][i, col]))
        known = ev.loc[ev.available_at.le(row.t)].sort_values('available_at').drop_duplicates('source_id', keep='last')
        current = known.loc[(known.measured_at - current_time).abs().le(pd.Timedelta(microseconds=1))]
        assert len(current) and current.value.eq(labels['current'][i, col]).all()
        future = ev.loc[ev.source_id.eq(labels['future_source_id'][i, col])].sort_values('available_at').iloc[-1]
        assert future.value == labels['future'][i, col]
        methods = set(current.measurement_method)
        strata[i] = ('ambiguous' if len(methods) != 1 else
                     'same' if future.measurement_method in methods else 'switched')
    summaries = {name: {'rows': int((strata == name).sum()),
                       'patients': int(q.loc[strata == name, 'subject_id'].nunique())}
                 for name in ('same', 'switched', 'ambiguous')}
    results = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        points = dict(np.load(root / f'bedside_map_predictions_{seed}.npz'))
        roles = dict(np.load(root / f'partition_{seed}.npz'))
        item = {'seed': seed, 'roles': {}}
        for role in ('evaluation', 'hospital'):
            item['roles'][role] = {}
            for name in summaries:
                mask = roles[role][valid] & (strata == name)
                f = q.loc[mask]
                item['roles'][role][name] = paired_model_errors(
                    f, {k: p[mask] for k, p in points.items()}, 'own',
                    seed=20260912, repetitions=20000)
        results.append(item)
    out.write_text(json.dumps({'complete': True, 'training_performed': False,
        'runtime_modified': False, 'future_strata_used_for_routing': False,
        'strata': summaries, 'seeds': results,
        'bindings': {str(p): sha256(p) for p in (parent, training, events_path, Path(__file__))}}, indent=2) + '\n')
    print(summaries)
    for role in ('evaluation', 'hospital'):
        for name in summaries:
            scores = [r['roles'][role][name]['bedside'] for r in results]
            print(role, name, 'patients', [s['patients'] for s in scores],
                  'gain', sum(s['ci95'] is not None and s['ci95'][1] < 0 for s in scores),
                  'harm', sum(s['ci95'] is not None and s['ci95'][0] > 0 for s in scores))


if __name__ == '__main__':
    main()

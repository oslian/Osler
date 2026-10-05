"""Independently replay E23 compartments and locate fit/selection tradeoffs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from scripts.probe_organ_compartments import DEST, write_json
from scripts.train_content_retention import build_fold
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_reconstruction_organ_compartments_replay_20261003.json')
    if output.exists(): raise ValueError('Replay verification already exists')
    ap = Path('artifacts/native_reconstruction_organ_compartments_assessment_20261003.json')
    assessment = json.loads(ap.read_text())
    vp = Path('artifacts/native_reconstruction_organ_compartments_verification_20261003.json')
    verification = json.loads(vp.read_text())
    assert verification['bindings'][str(ap)] == sha256(ap)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    records, errors, contexts = [], [], 0
    bindings = {str(ap): sha256(ap), str(vp): sha256(vp), str(Path(__file__)): sha256(Path(__file__))}
    with torch.no_grad():
        for fold in range(5):
            variables, examples, model, parent, _, _ = build_fold(fold)
            report_path = Path(f'artifacts/native_reconstruction_organ_compartments_fold{fold}_20261003.json')
            assert assessment['bindings'][str(report_path)] == sha256(report_path)
            report = json.loads(report_path.read_text()); layer = model.trajectory.layer
            assert report['contract']['encoded_contract'] == parent
            for p, h in report['contract']['bindings'].items():
                assert sha256(Path(p)) == h
            for role, rows in examples.items():
                path = DEST/f'fold{fold}/{role}.joblib'
                assert sha256(path) == report['extraction'][role]['sha256']
                data = joblib.load(path)
                for i, e in enumerate(rows):
                    b, o, presence = (e[k] for k in ('body', 'organs', 'presence'))
                    initial = layer.initialize(b.new_zeros((1, layer.body_dim)),
                        b.new_zeros((1, layer.organ_count, layer.organ_dim)),
                        b.new_zeros((1, layer.organ_count)))
                    times = (e['times']-e['times'][0]).to(b.dtype)
                    state = layer.replay_observations(initial, b[None], o[None], presence[None], times[None])
                    gap = (e['times'].new_tensor(e['anchor'])-e['times'][-1]).to(b.dtype).reshape(1)
                    state = layer.advance(state, gap)
                    native = float(model.readout(state.body)[0, e['target']])
                    np.testing.assert_array_equal(state.body[0].numpy(), data['body'][i])
                    np.testing.assert_array_equal(state.organs[0].numpy().reshape(-1), data['organs'][i])
                    assert native == data['native'][i]
                    errors.append(float(np.max(abs(state.organs[0].numpy().reshape(-1)-data['organs'][i]))))
                    contexts += 1
            for v, info in report['targets'].items():
                if info['status'] != 'fitted': continue
                pp = DEST/f'fold{fold}/{v}.parquet'
                assert sha256(pp) == info['prediction_sha256']
                frame = pd.read_parquet(pp)
                selection = {r+'_'+k: float(np.mean(abs(frame[r+'_'+k]-frame.truth)))
                             for r in ('ridge', 'hgb') for k in ('body', 'organs', 'raw', 'shuffled', 'expanded')}
                records.append(dict(fold=fold, variable=v, fit_patients=info['support']['fit'],
                    selection_patients=info['support']['selection'], fit=info['fit_mae'], selection=selection))
            print('Independent full compartment replay', fold, flush=True)
    diagnostics = {}
    for r in ('ridge', 'hgb'):
        fit_better = [x for x in records if x['fit'][r+'_organs'] < x['fit'][r+'_body']]
        diagnostics[r] = dict(variable_folds=len(records), organs_better_on_fit=len(fit_better),
            organs_better_on_selection=sum(x['selection'][r+'_organs'] < x['selection'][r+'_body'] for x in records),
            fit_better_but_selection_worse=sum(x['selection'][r+'_organs'] > x['selection'][r+'_body'] for x in fit_better),
            organs_fit_not_better=len(records)-len(fit_better))
    support = {key: dict(min=int(min(x[key] for x in records)), median=float(np.median([x[key] for x in records])),
                        max=int(max(x[key] for x in records))) for key in ('fit_patients', 'selection_patients')}
    result = dict(contexts=contexts, organ_max_error=max(errors), body_and_native_exact=True,
        support=support, descriptive_fit_selection=diagnostics)
    write_json(output, dict(summary=result, diagnostics=records, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Paired target-fold means locate this bounded reader failure; not independent replications or proof of information absence.'))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()

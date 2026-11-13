"""Replay fixed correction fits to separate fitting-objective and transfer harm."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.try_conditional_reconstruction_repair import ARMS, predict, weights
from whole_body_jepa.multisource_intervals import sha256


def losses(frame, predictions):
    w = weights(frame)
    return {a: dict(mae=float(np.average(abs(frame.truth.to_numpy()-p), weights=w)),
                    mse=float(np.average((frame.truth.to_numpy()-p)**2, weights=w)),
                    mean_signed_error=float(np.average(p-frame.truth.to_numpy(), weights=w)))
            for a, p in predictions.items()}


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_conditional_repair_20260925.json')
    out = Path('artifacts/native_reconstruction_conditional_fit_objective_20260925.json')
    if out.exists():
        raise ValueError('Existing fit diagnostic')
    r = json.loads(rp.read_text())
    for p, h in r['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed correction dependency')
    private = root/'conditional_repair_20260925.parquet'
    if sha256(private) != r['private_sha256']:
        raise ValueError('Changed predictions')
    expected = pd.read_parquet(private)
    variables = json.loads(Path('artifacts/native_reconstruction_variable_snapshot_20260925.json').read_text())['variables']
    cal = {f: pd.read_parquet(root/f'calibration_20260924/fold{f}.parquet') for f in range(5)}
    original = {f: pd.read_parquet(root/f'outer_20260924/fold{f}.parquet') for f in range(5)}
    results = {}
    with threadpool_limits(limits=4):
        for t in r['targets']:
            snap = pd.read_parquet(root/'variable_snapshot_20260925.parquet', filters=[('target', '=', t)])
            results[t] = {}
            for f in range(5):
                source = cal[f].loc[cal[f].target == t].merge(snap, on=['patient', 'target'], validate='one_to_one')
                outer = original[f].loc[original[f].target == t].merge(snap, on=['patient', 'target'], validate='one_to_one')
                query = pd.concat([source, outer], ignore_index=True)
                p, info = predict(source, query, variables)
                check = expected.loc[(expected.target == t) & (expected.fold == f)]
                if check.patient.tolist() != outer.patient.tolist():
                    raise ValueError('Replay identity mismatch')
                for a in ARMS:
                    if not np.allclose(p[a][len(source):], check[a], atol=1e-10, rtol=1e-12):
                        raise ValueError('Fixed prediction did not reproduce')
                if info['status'] != 'fitted' or source.empty:
                    results[t][str(f)] = dict(status='insufficient_support')
                    continue
                results[t][str(f)] = dict(status='fitted', source_fit=losses(source, {a:v[:len(source)] for a,v in p.items()}),
                    outer=losses(outer, {a:v[len(source):] for a,v in p.items()}) if len(outer) else None)
    assessed = [v for f in results.values() for v in f.values() if v['status'] == 'fitted']
    summary = dict(fitted=len(assessed), outer_predictions_reproduced=True,
        source_fit_mse_improved_but_mae_worse=sum(v['source_fit']['numeric_context']['mse'] < v['source_fit']['unchanged']['mse'] and
            v['source_fit']['numeric_context']['mae'] > v['source_fit']['unchanged']['mae'] for v in assessed),
        source_fit_mae_worse=sum(v['source_fit']['numeric_context']['mae'] > v['source_fit']['unchanged']['mae'] for v in assessed))
    out.write_text(json.dumps(dict(summary=summary, targets=results,
        bindings={str(p):sha256(p) for p in (rp, private, Path(__file__))},
        runtime_modified=False, promotion_allowed=False,
        limitation='Posthoc in-sample replay diagnoses this fixed squared-loss correction only, not generalization or a unique cause of past JEPA failures. No objective/hyperparameter retuning.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Descriptive error localization, never a truth-conditioned deployment gate."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256


def main():
    out = Path('artifacts/native_reconstruction_site_error_shape_20260917.json')
    if out.exists():
        raise ValueError('existing report')
    evaluation = Path('artifacts/native_reconstruction_site_evaluation_20260917.json')
    report = json.loads(evaluation.read_text())
    for p, digest in report['bindings'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('changed evaluation binding')
    saved = Path('data/research_recovery/site_reconstruction_training_20260917/evaluation_predictions.pt')
    rows = torch.load(saved, weights_only=False)
    split = Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    roles = torch.load(split, weights_only=False)['roles']
    index_path = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt')
    index = torch.load(index_path, weights_only=False)
    fit = {t: [] for t in report['targets']}
    for shard in index['shards']:
        if roles[shard['patient']] != 'fit':
            continue
        path = durable(shard['path'])
        if sha256(path) != shard['sha256']:
            raise ValueError('changed source shard')
        data = torch.load(path, weights_only=False)
        for c, example in data['samples'].items():
            target = index['variables'][c]
            if target in fit:
                fit[target].append(example['truth_normalized'])
    results = {}
    for target, original in report['targets'].items():
        part = [r for r in rows if r['target'] == target]
        if len(part) != original['patients'] or len({r['subject_id'] for r in part}) != len(part):
            raise ValueError('support/identity mismatch')
        if any(roles[r['subject_id']] != 'evaluation' for r in part):
            raise ValueError('wrong data role')
        truth = np.array([r['truth'] for r in part])
        pred = {a: np.array([r[a] for r in part]) for a in original['normalized_mae']}
        if not np.isfinite(truth).all() or any(not np.isfinite(p).all() for p in pred.values()):
            raise ValueError('nonfinite input')
        error = {a: abs(p-truth) for a, p in pred.items()}
        for a, e in error.items():
            if not np.isclose(e.mean(), original['normalized_mae'][a], atol=1e-10, rtol=0):
                raise ValueError('MAE mismatch')
        center = float(np.median(fit[target]))
        radius = float(np.quantile(abs(np.asarray(fit[target])-center), .90))
        tail = abs(truth-center) > radius
        delta = error['real']-error['process']
        strata = {}
        for name, mask in (('central', ~tail), ('tail', tail)):
            strata[name] = dict(patients=int(mask.sum()),
                mean_real_minus_process=None if not mask.any() else float(delta[mask].mean()),
                contribution_to_total_mean=float(delta[mask].sum()/len(delta)))
        results[target] = dict(patients=len(part), fit_center=center, fit_p90_distance=radius,
            real_better_fraction=float(np.mean(delta < 0)), real_worse_fraction=float(np.mean(delta > 0)),
            median_paired_error_delta=float(np.median(delta)), strata=strata,
            models={a: dict(mae=float(error[a].mean()), median_absolute_error=float(np.median(error[a])),
                mean_signed_error=float(np.mean(p-truth)), median_signed_error=float(np.median(p-truth)),
                spearman=None if np.ptp(p)==0 or np.ptp(truth)==0 else float(spearmanr(p, truth).statistic))
                for a,p in pred.items()})
    paths = [Path(__file__), evaluation, saved, split, index_path]
    out.write_text(json.dumps(dict(complete=True, targets=results, unsupported=report['unsupported'],
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Retrospective truth strata using source-fit thresholds; unavailable at prediction time. '
        'No exclusions, fitting, gate, multiple-testing claims or causal attribution. '
        'Correlations and conditional error summaries are descriptive.'), indent=2)+'\n')
    for target in ('lipase', 'triglycerides', 'bilirubin', 'glucose', 'creatinine'):
        r = results[target]
        print(target, 'win fraction', r['real_better_fraction'], 'median paired', r['median_paired_error_delta'], 'strata', r['strata'])


if __name__ == '__main__':
    main()

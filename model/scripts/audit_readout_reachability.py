"""Audit algebraic reachability of saved readouts, not physiological efficacy."""
import json
from pathlib import Path

import numpy as np
import torch

from whole_body_jepa.multisource_intervals import sha256


ROOT = Path('data/research_recovery/additional_development_domains_20260921')
SOURCE = Path('artifacts/native_actual_step_20261005.json')
SCREEN = Path('artifacts/native_screen_failures_20261005.json')
OUTPUT = Path('artifacts/native_readout_reachability_20261005.json')


def matrix_audit(weight):
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 2 or not w.size or not np.isfinite(w).all():
        raise ValueError('Finite nonempty readout matrix required')
    s = np.linalg.svd(w, compute_uv=False)
    tolerance = float(max(w.shape) * np.finfo(float).eps * s[0])
    rank = int(np.sum(s > tolerance))
    inverse = np.linalg.pinv(w, rcond=max(w.shape)*np.finfo(float).eps)
    projection = w @ inverse
    # This probes prescribed output-coordinate changes, not patient targets.
    error64 = float(np.max(np.abs(projection - np.eye(len(w)))))
    error32 = float(np.max(np.abs(w.astype(np.float32) @ inverse.astype(np.float32)
                                 - np.eye(len(w)))))
    return dict(shape=list(w.shape), rank=rank, rank_tolerance=tolerance,
                zero_rows=np.flatnonzero(~np.any(w != 0, axis=1)).tolist(),
                full_row_rank=rank == len(w), singular_values=s.tolist(),
                condition_number=float(s[0]/s[-1]) if rank == min(w.shape) else None,
                output_projector_error64=error64, output_projector_error32=error32,
                minimum_state_norm_per_unit_output=np.linalg.norm(inverse, axis=0).tolist())


def audit_targets(weight, bias, variables, statuses):
    w, b = np.asarray(weight), np.asarray(bias)
    if (w.ndim != 2 or w.shape[0] != len(variables) or b.shape != (len(variables),)
            or len(set(variables)) != len(variables) or set(variables) != set(statuses)
            or not np.isfinite(b).all()):
        raise ValueError('Readout and complete target inventory must match')
    supported = {}
    for v in variables:
        record = statuses[v]
        if record['comparisons']:
            supported[v] = record['criteria']['support']
            if not isinstance(supported[v], bool):
                raise ValueError('Boolean support required')
        elif record.get('first_unmet') == 'unsupported':
            supported[v] = False
        else:
            raise ValueError('Unscored target must explicitly be unsupported')
    subsets = dict(active=np.flatnonzero(np.any(w != 0, axis=1)).tolist(),
                   scored=[i for i, v in enumerate(variables) if statuses[v]['comparisons']],
                   supported=[i for i, v in enumerate(variables)
                              if supported[v]])
    zeros = np.flatnonzero(~np.any(w != 0, axis=1)).tolist()
    return dict(all=matrix_audit(w),
                zero_heads=[dict(target=variables[i], bias=float(b[i]),
                                 scored=bool(statuses[variables[i]]['comparisons']),
                                 supported=supported[variables[i]])
                            for i in zeros],
                subsets={key: dict(targets=[variables[i] for i in indices],
                                   matrix=matrix_audit(w[indices]) if indices else None)
                         for key, indices in subsets.items()})


def main():
    if OUTPUT.exists():
        raise ValueError('Completed audit exists; do not overwrite')
    source = json.loads(SOURCE.read_text())
    screen = json.loads(SCREEN.read_text())
    if not source['complete'] or not screen['complete']:
        raise ValueError('Completed source audits required')
    if screen['bindings'][str(SOURCE)] != sha256(SOURCE):
        raise ValueError('Changed E40 source')
    bound = [SOURCE, SCREEN, Path(__file__).relative_to(Path.cwd()),
             Path('tests/test_readout_reachability.py'),
             Path('RECONSTRUCTION_READOUT_REACHABILITY_PROTOCOL_20261005.md')]
    rows = []
    for fold in range(5):
        cp = ROOT/f'actual_step_20261005/fold{fold}/contract.json'
        wp = cp.parent/'actual_step.pt'
        for p in (cp, wp):
            if source['bindings'].get(str(p)) != sha256(p):
                raise ValueError('Changed saved contract or checkpoint')
        contract = json.loads(cp.read_text())
        variables = contract['parent']['variables']
        d = torch.load(wp, weights_only=False, map_location='cpu')
        w, b = (d['model.readout.'+key].double().numpy() for key in ('weight', 'bias'))
        result = audit_targets(w, b, variables, screen['targets'])
        # Independent torch implementation checks NumPy rank and pseudoinverse.
        tw = torch.from_numpy(w)
        if result['all']['rank'] != int(torch.linalg.matrix_rank(tw)):
            raise ValueError('Independent rank disagreement')
        ti = torch.linalg.pinv(tw)
        np.testing.assert_allclose((tw @ ti).numpy(), w @ np.linalg.pinv(w), atol=1e-10, rtol=0)
        rows.append(dict(fold=fold, checkpoint_sha256=sha256(wp), **result))
        bound.extend((cp, wp))
    report = dict(complete=True, folds=rows,
                  bindings={str(p): sha256(p) for p in bound},
                  training_performed=False, runtime_modified=False, promotion_allowed=False,
                  limitation='Full row rank only excludes a hard affine output dependency. '
                  'It does not prove the recurrent encoder can infer or reach these states, '
                  'nor that noisy or missing targets are predictable. Pseudoinverse unit '
                  'changes are algebraic diagnostics, never runtime interventions. '
                  'Fold0/2 are not independent replications; reused development only.')
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps([dict(fold=r['fold'], zero_heads=r['zero_heads'],
                         active_rank=r['subsets']['active']['matrix']['rank'],
                         supported_rank=r['subsets']['supported']['matrix']['rank'],
                         active_condition=r['subsets']['active']['matrix']['condition_number'])
                     for r in rows], indent=2))


if __name__ == '__main__':
    main()

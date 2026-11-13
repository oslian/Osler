"""Source-only numerical readout repair with matched process calibration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
from scipy import sparse

from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def fit_affine(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('aligned finite nonempty calibration required')
    if np.ptp(x) == 0:
        return 0., float(np.median(y))
    n = len(x)
    design = sparse.csr_matrix(np.column_stack((x, np.ones(n))))
    identity = sparse.eye(n, format='csr')
    constraints = sparse.vstack((sparse.hstack((design, -identity)), sparse.hstack((-design, -identity))))
    result = linprog(np.r_[0., 0., np.ones(n)/n], A_ub=constraints,
                     b_ub=np.r_[y, -y], bounds=[(0., None), (None, None)]+[(0., None)]*n,
                     method='highs')
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('calibration optimization failed')
    return float(result.x[0]), float(result.x[1])


def load_saved(report, filename, bindings):
    matches = [(durable(p), h) for p, h in report['bindings'].items() if Path(p).name == filename]
    if len(matches) != 1 or sha256(matches[0][0]) != matches[0][1]:
        raise ValueError('prediction identity mismatch')
    path, digest = matches[0]
    bindings[str(path)] = digest
    return torch.load(path, weights_only=False)


def main():
    out = Path('artifacts/native_reconstruction_readout_calibration_20260917.json')
    if out.exists():
        raise ValueError('existing experiment')
    protocol = Path('RECONSTRUCTION_READOUT_CALIBRATION_PROTOCOL_20260917.md')
    bindings = {str(p): sha256(p) for p in (Path(__file__), protocol, Path('whole_body_jepa/change_evaluation.py'))}
    results = {}
    for seed in (19, 31, 43, 59, 71, 83):
        sp = Path(f'artifacts/native_reconstruction_source_fit_{seed}_20260917.json')
        ep = Path(f'artifacts/native_reconstruction_replication_v2_{seed}_evaluation_20260914.json')
        source, evaluation = json.loads(sp.read_text()), json.loads(ep.read_text())
        bindings.update({str(p): sha256(p) for p in (sp, ep)})
        raw = load_saved(source, 'source_predictions.pt', bindings)
        held = load_saved(evaluation, 'heldout_predictions.pt', bindings)
        frames = {r: pd.DataFrame(rows) for r, rows in (('fit',raw['fit']),('tune',raw['tune']),('evaluation',held['evaluation']))}
        for role, frame in frames.items():
            if frame.duplicated(['subject_id','target']).any():
                raise ValueError('duplicate identity')
            for other, part in frames.items():
                if role != other and set(frame.subject_id) & set(part.subject_id):
                    raise ValueError('patient role overlap')
        targets = {}
        for target, part in frames['evaluation'].groupby('target'):
            tune = frames['tune'].loc[frames['tune'].target == target]
            if len(tune) < 30:
                raise ValueError('source support changed')
            for role, table in (('tune', tune), ('evaluation', part)):
                old = (source if role == 'tune' else evaluation)['roles'][role][target]['normalized_mae']
                for arm in ('real','process','population'):
                    if not np.isclose(abs(table[arm]-table.truth).mean(), old[arm], atol=1e-10, rtol=0):
                        raise ValueError('score reproduction failed')
            pred = {a:part[a].to_numpy() for a in ('real','process','population')}
            coefficients = {}
            for arm in ('real','process'):
                shift = float(np.median(tune.truth-tune[arm]))
                a,b = fit_affine(tune[arm], tune.truth)
                pred[arm+'_offset'] = pred[arm]+shift
                pred[arm+'_affine'] = a*pred[arm]+b
                coefficients[arm] = dict(offset=shift, slope=a, intercept=b)
            comparisons = {reference:paired_model_errors(part,pred,reference,seed=20260917,repetitions=2000)
                           for reference in ('real','population','process_offset','process_affine')}
            targets[target] = dict(patients=len(part), coefficients=coefficients,
                mae={a:float(np.mean(abs(p-part.truth.to_numpy()))) for a,p in pred.items()}, comparisons=comparisons)
        results[str(seed)] = dict(targets=targets, unsupported=source['source_unsupported'])
        print('complete source calibration', seed, flush=True)
    out.write_text(json.dumps(dict(complete=True,seeds=results,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='Previously inspected source development data, tune-fitted calibration; nominal uncorrected CIs and overlapping splits. No hospital/lockbox or deployment claim.'),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

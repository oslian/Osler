"""Verify an extracted public bundle without private datasets or repo imports."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import transform_example
from scripts.load_coverage_candidate import load
from whole_body_jepa.contracts import VARIABLE_UNITS
from whole_body_jepa.coverage_input import build_input
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    manifest = json.loads(Path('MANIFEST.json').read_text())
    for name, info in manifest['files'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Bundle hash mismatch: ' + name)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    results = {}
    for fold in (0, 1, 3, 4):
        name = f'coverage_fold{fold}_real_epoch10.pt'
        model, checkpoint = load(name)
        variables = checkpoint['variables']
        n = len(variables)
        scaler = {k: v.numpy() for k, v in checkpoint['scaler'].items()}
        rows = [dict(stay_id='synthetic', source_id=f'synthetic-{i}', source_kind='lab_available',
                     event_time='2100-01-01 01:00:00', measured_at='2100-01-01 00:40:00',
                     variable=v, value=float(scaler['location'][i]), unit=VARIABLE_UNITS[v], quality_valid=True)
                for i, v in enumerate(variables) if scaler['counts'][i] > 0]
        events = pd.DataFrame(rows)
        queries = pd.DataFrame([dict(stay_id='synthetic', t='2100-01-01 01:00:00')])
        reference = {x['target_index']: x for x in reconstruction_examples(events, queries, variables,
                     np.zeros(n), np.ones(n), checkpoint['schema']['state_dict']['variable_module_ids'].numpy(),
                     len(checkpoint['schema']['modules']))}
        checked = []
        with torch.inference_mode():
            for target in checkpoint['targets']:
                col = variables.index(target)
                old = transform_example(reference[col], scaler)
                new = build_input(events, checkpoint,
                                  'synthetic', '2100-01-01 01:00:00', target)
                missing = build_input(events.loc[events.variable != target], checkpoint,
                                      'synthetic', '2100-01-01 01:00:00', target)
                for key in ('tokens', 'event_hours'):
                    if not torch.equal(old[key], new[key]):
                        raise ValueError('Training/input tensor mismatch')
                if old['anchor_hour'] != new['anchor_hour']:
                    raise ValueError('Anchor mismatch')
                a = model(old['tokens'], old['event_hours'], old['anchor_hour'], target)
                b = model(new['tokens'], new['event_hours'], new['anchor_hour'], target)
                z = model(missing['tokens'], missing['event_hours'], missing['anchor_hour'], target)
                # Redundant module-presence fields may differ when an input row
                # is omitted; the model recomputes them from the masked values.
                if (not torch.equal(old['tokens'][:, :3*n], missing['tokens'][:, :3*n])
                        or not torch.equal(old['event_hours'], missing['event_hours'])
                        or not torch.equal(a, b) or not torch.equal(a, z)
                        or not torch.isfinite(b).all()):
                    raise ValueError('Input prediction parity failed')
                checked.append(target)
        results[str(fold)] = dict(targets=checked, exact_tensor_and_prediction_parity=True,
                                 omitting_target_row_preserves_prediction=True)
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in ('scripts', 'whole_body_jepa', 'osler_jepa', 'whole_body_joint_jepa'):
            file = getattr(module, '__file__', None)
            if file is not None and not Path(file).resolve().is_relative_to(root):
                raise ValueError('Project module imported outside extracted bundle: ' + name)
    report = dict(folds=results, checks=sum(len(r['targets']) for r in results.values()),
                  assets_sha256_match=True, project_imports_within_bundle=True,
                  real_patient_data_used=False, runtime_modified=False, promotion_allowed=False,
                  limitation='Synthetic packaging, transformation and weight-loading parity only; not clinical accuracy or external validation.')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('Complete bundle parity:', report['checks'], 'fold-target checks')


if __name__ == '__main__':
    main()

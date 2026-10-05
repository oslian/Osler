"""Descriptive paired selection summary; never a promotion gate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    plan = torch.load(root / 'partitions_20260922.pt', weights_only=False)
    arms = ('real', 'process', 'replacement_real', 'replacement_process', 'population')
    records, bindings = [], {}
    for fold in range(5):
        path = root / f'coverage_selection_20260926/fold{fold}.parquet'
        meta = json.loads(path.with_suffix('.json').read_text())
        digest = sha256(path)
        if digest != meta['prediction_sha256']:
            raise ValueError('Prediction hash mismatch')
        bindings[str(path)] = digest
        frame = pd.read_parquet(path)
        if frame.duplicated(['patient', 'target']).any() or not frame.fold.eq(fold).all():
            raise ValueError('Invalid paired identity')
        if not all(plan['folds'][fold][p] == 'selection' for p in frame.patient.unique()):
            raise ValueError('Wrong patient role')
        if not np.isfinite(frame[[*arms, 'truth', 'fit_scale']]).all().all() or not frame.fit_scale.gt(0).all():
            raise ValueError('Invalid numeric output')
        for arm in arms:
            frame[arm + '_error'] = abs(frame[arm] - frame.truth) / frame.fit_scale
        columns = [arm + '_error' for arm in arms]
        # One query per patient-target, verified above: rows are patient-equal here.
        patient = frame.groupby('target')[columns].mean()
        component = frame.groupby(['target', 'component'])[columns].mean().groupby('target').mean()
        records.append(dict(
            fold=fold, pairs=len(frame), patients=int(frame.patient.nunique()),
            targets=len(patient), patient_macro=patient.mean().to_dict(),
            component_macro=component.mean().to_dict(),
            patient_target_errors=patient.to_dict(orient='index'),
            component_target_errors=component.to_dict(orient='index'),
            targets_patient_improved=int((patient.real_error < patient.replacement_real_error).sum()),
            targets_component_improved=int((component.real_error < component.replacement_real_error).sum())))
    result = dict(folds=records, bindings=bindings, promotion_allowed=False,
                  limitation='Descriptive selection comparison. Repeated query patients and overlapping fits are not independent replications. Outer evaluation pending.')
    dest = Path('artifacts/native_reconstruction_coverage_selection_summary_20260927.json')
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if 'target_errors' not in k} for r in records], indent=2))


if __name__ == '__main__':
    main()

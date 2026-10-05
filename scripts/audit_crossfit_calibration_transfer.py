"""Retrospective calibration-transfer diagnosis; oracle is never deployed."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.audit_crossfit_offset_calibration import fit_offset
from whole_body_jepa.multisource_intervals import sha256


def compare(frame, calibration):
    oracle = fit_offset(frame)
    original = (frame.real-frame.truth).abs()
    applied = (frame.real+calibration['offset']-frame.truth).abs()
    supported = oracle['status'] == 'calibrated'
    optimum = (frame.real+oracle['offset']-frame.truth).abs() if supported else None
    # Use exactly the equal-group objective used to fit the median offset.
    def objective(v):
        return float(v.groupby(frame.group).mean().mean())
    loss = dict(original=objective(original), applied=objective(applied),
                retrospective_oracle=objective(optimum) if supported else None)
    if supported and loss['retrospective_oracle'] > min(loss['original'], loss['applied'])+1e-10:
        raise ValueError('Median oracle does not minimize its stated objective')
    return dict(patients=len(frame), calibration=calibration, retrospective_oracle=oracle,
        equal_group_mae=loss,
        supported_both=bool(supported and calibration['status']=='calibrated'),
        opposite_direction=bool(calibration['offset']*oracle['offset'] < 0) if supported else None,
        patient_delta=float((applied-original).mean()),
        equal_group_delta=loss['applied']-loss['original'])


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_crossfit_offset_calibration_20260921.json')
    out = Path('artifacts/native_reconstruction_crossfit_calibration_transfer_20260921.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    parent = json.loads(rp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed calibration dependency')
    ep = Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    outer = json.loads(ep.read_text())
    records = {}; paths = [Path(__file__), rp, ep]
    for fold in range(5):
        p = root/'encoder_crossfit_outer_20260921'/f'fold{fold}.parquet'
        if sha256(p) != outer['prediction_hashes'][str(fold)]:
            raise ValueError('Changed prediction cache')
        calpath = root/'encoder_crossfit_offset_calibration_20260921'/f'calibration{fold}.parquet'
        if sha256(calpath) != parent['calibration_hashes'][str(fold)]:
            raise ValueError('Changed calibration cache')
        cal = pd.read_parquet(calpath); frame = pd.read_parquet(p)
        if set(cal.patient) & set(frame.patient):
            raise ValueError('Calibration/test overlap')
        records[str(fold)] = {}
        for target in parent['targets']:
            source = cal.loc[cal.target.eq(target) & cal.status.eq('ok')]
            calibration = fit_offset(source)
            if calibration != parent['calibration'][str(fold)][target]:
                raise ValueError('Source calibration not reproduced')
            part = frame.loc[frame.target.eq(target) & frame.status.eq('ok')]
            records[str(fold)][target] = compare(part, calibration)
        paths.extend([p, calpath])
    eligible = [v for r in records.values() for v in r.values() if v['supported_both']]
    summary = dict(fold_targets=165, supported_both=len(eligible),
        opposite_direction=sum(v['opposite_direction'] for v in eligible),
        applied_equal_group_harm=sum(v['equal_group_delta'] > 0 for v in eligible),
        applied_patient_harm=sum(v['patient_delta'] > 0 for v in eligible))
    out.write_text(json.dumps(dict(summary=summary, folds=records,
        bindings={str(p):sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Retrospective diagnostic oracle uses outer labels; forbidden for deployment, model selection or calibration. Same fixed predictions, no model fitting. Calibration domains thin; not proof of unique biological cause.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

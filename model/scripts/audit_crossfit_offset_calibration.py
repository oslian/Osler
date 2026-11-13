"""Source-calibration-only intercept repair of frozen fresh encoders."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.train_encoder_crossfit import fresh_encoder
from scripts.train_joint_source_reconstruction import equal_group_median
from scripts.encoder_fold_support import supported_example
from scripts.encoder_fold_scaling import inverse_value
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256


def fit_offset(frame):
    groups = [frame.loc[frame.group.eq(g), 'truth'].to_numpy()-
              frame.loc[frame.group.eq(g), 'real'].to_numpy() for g in ('selected', 'broad')]
    counts = dict(zip(('selected', 'broad'), map(len, groups)))
    if min(counts.values()) < 30:
        return dict(offset=0., status='insufficient_calibration', counts=counts)
    if not all(np.isfinite(g).all() for g in groups):
        raise ValueError('Nonfinite calibration residual')
    return dict(offset=equal_group_median(groups), status='calibrated', counts=counts)


def summarize(frame):
    checks = {}; err = (frame.corrected-frame.truth).abs()
    for arm in ('real', 'process', 'population'):
        delta = err-(frame[arm]-frame.truth).abs()
        checks[arm] = evidence(delta.to_numpy(), frame.component.to_numpy())
    return dict(patients=len(frame), calibrated_patients=int(frame.calibration_status.eq('calibrated').sum()),
        mae={a:float((frame[a]-frame.truth).abs().mean()) for a in ('corrected', 'real', 'process', 'population')},
        comparisons=checks, descriptive_screen_pass=confirmed(checks, len(frame)))


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    output = Path('artifacts/native_reconstruction_crossfit_offset_calibration_20260921.json')
    if output.exists():
        raise ValueError('Existing experiment')
    parent = json.loads(rp.read_text())
    for p, h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed outer dependency')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    sp = root/'encoder_fold_scaling_20260920.pt'
    jp = root/'joint_source_manifest.pt'
    cp = Path('models/whole_body_joint_jepa.pt')
    plan = torch.load(fp, weights_only=False)
    scalers = torch.load(sp, weights_only=False)['scalers']
    schema = torch.load(jp, weights_only=False)
    variables, targets = schema['variables'], schema['targets']
    cols = {variables.index(t) for t in targets}
    model_schema = torch.load(cp, map_location='cpu', weights_only=False)
    dependencies = [Path(__file__), rp, fp, sp, jp, cp,
        Path('RECONSTRUCTION_CROSSFIT_OFFSET_PROTOCOL_20260921.md')]
    bindings = {str(p): sha256(p) for p in dependencies}
    folder = root/'encoder_crossfit_offset_calibration_20260921'; folder.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    outer = []; calibrations = {}; hashes = {}
    for fold, roles in enumerate(plan['folds']):
        path = folder/f'calibration{fold}.parquet'; meta = folder/f'calibration{fold}.json'
        if path.exists() and meta.exists():
            info = json.loads(meta.read_text())
            if info['bindings'] != bindings or sha256(path) != info['sha256']:
                raise ValueError('Changed calibration cache')
            cal = pd.read_parquet(path)
        else:
            ck = torch.load(root/'encoder_crossfit_training_20260920'/f'fold{fold}_real.pt', weights_only=False)
            model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(model_schema), variables, ('map',))).eval()
            model.load_state_dict(ck['state_dict'], strict=True)
            rows = []
            with torch.inference_mode():
                for shard in plan['shards']:
                    person = shard['patient']
                    if roles[person] != 'encoder_tune':
                        continue
                    p = Path(shard['path'])
                    if sha256(p) != shard['sha256']:
                        raise ValueError('Changed shard')
                    for c, raw in torch.load(p, weights_only=False)['samples'].items():
                        if c not in cols:
                            continue
                        row = dict(patient=person, target=variables[c], group=shard['group'],
                                   truth=raw['truth_normalized'], real=np.nan, status='ok')
                        try:
                            e = supported_example(raw, scalers[fold])
                        except ValueError as exc:
                            if str(exc) not in ('No source-supported independent history remains', 'Target has no source normalization support'):
                                raise
                            row['status'] = str(exc); rows.append(row); continue
                        value = float(model(e['tokens'], e['event_hours'], e['anchor_hour'], variables[c])[0])
                        row['real'] = float(inverse_value(value, c, scalers[fold]))
                        if not np.isfinite(row['real']):
                            raise ValueError('Nonfinite calibration prediction')
                        rows.append(row)
            cal = pd.DataFrame(rows)
            if cal.duplicated(['patient', 'target']).any():
                raise ValueError('Duplicate calibration pair')
            tmp = path.with_suffix('.tmp'); cal.to_parquet(tmp, index=False); tmp.replace(path)
            meta.write_text(json.dumps(dict(bindings=bindings, sha256=sha256(path)), indent=2)+'\n')
        hashes[str(fold)] = sha256(path)
        if any(roles[p] != 'encoder_tune' for p in cal.patient.unique()):
            raise ValueError('Calibration role leak')
        # These offsets are completely determined before reading outer labels.
        offsets = {t:fit_offset(cal.loc[cal.target.eq(t) & cal.status.eq('ok')]) for t in targets}
        calibrations[str(fold)] = offsets
        p = root/'encoder_crossfit_outer_20260921'/f'fold{fold}.parquet'
        if sha256(p) != parent['prediction_hashes'][str(fold)]:
            raise ValueError('Changed outer predictions')
        test = pd.read_parquet(p)
        if set(test.patient) & set(cal.patient):
            raise ValueError('Calibration/test overlap')
        test['corrected'] = test.real+test.target.map({t:v['offset'] for t,v in offsets.items()})
        test['calibration_status'] = test.target.map({t:v['status'] for t,v in offsets.items()})
        outer.append(test)
        print('Completed calibration fold', fold, 'pairs', len(cal), flush=True)
    full = pd.concat(outer, ignore_index=True)
    if full.duplicated(['patient', 'target']).any() or len(full) != parent['summary']['pairs']:
        raise ValueError('Outer coverage changed')
    good = full.loc[full.status.eq('ok')]
    result = {}
    for t, part in good.groupby('target'):
        result[t] = {s:summarize(part if s == 'all' else part.loc[part.group.eq(s)])
                     for s in ('all', 'selected', 'broad')}
    summary = dict(pairs=len(full), eligible_pairs=len(good), targets=len(result),
        passes={s:[t for t,r in result.items() if r[s]['descriptive_screen_pass']] for s in ('all', 'selected', 'broad')},
        macro={s:{a:float(np.mean([r[s]['mae'][a] for r in result.values()]))
                  for a in ('corrected', 'real', 'process', 'population')} for s in ('all', 'selected', 'broad')})
    output.write_text(json.dumps(dict(summary=summary, targets=result, calibration=calibrations,
        calibration_hashes=hashes, bindings=bindings, runtime_modified=False, promotion_allowed=False,
        limitation='Source-calibration-only fixed intercept repair. Design motivated by previously inspected outer outcomes; descriptive development, not new external validation. No ranking repair or runtime promotion.'), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

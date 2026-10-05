"""Frozen numeric-history sensitivity, never outcome-selected routing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.train_encoder_crossfit import fresh_encoder
from scripts.encoder_fold_support import supported_example
from scripts.encoder_fold_scaling import inverse_value
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256


INPUTS = ('bicarbonate', 'potassium', 'hemoglobin')


def numeric_ablation(tokens, column):
    result = tokens.clone()
    result[:, column] = 0
    return result


def scores(frame, arm):
    base = (frame.real-frame.truth).abs()
    error = (frame[arm]-frame.truth).abs()
    difference = error-base
    return dict(patients=len(frame), mae=float(error.mean()),
        delta_vs_real=float(difference.mean()),
        component_equal_delta=float(difference.groupby(frame.component).mean().mean()),
        delta_vs_process=float((error-(frame.process-frame.truth).abs()).mean()),
        delta_vs_population=float((error-(frame.population-frame.truth).abs()).mean()),
        observed_input_patients=int(frame[f'{arm}_observed'].sum()),
        prediction_change_mean=float((frame[arm]-frame.real).abs().mean()))


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    output = Path('artifacts/native_reconstruction_crossfit_numeric_ablation_20260921.json')
    if output.exists():
        raise ValueError('Existing diagnostic')
    rp = Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    parent = json.loads(rp.read_text())
    for p, h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed outer dependency')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    sp = root/'encoder_fold_scaling_20260920.pt'
    jp = root/'joint_source_manifest.pt'
    plan = torch.load(fp, weights_only=False)
    scalers = torch.load(sp, weights_only=False)['scalers']
    variables = torch.load(jp, weights_only=False)['variables']
    target = 'creatinine'; col = variables.index(target); n = len(variables)
    cp = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(cp, map_location='cpu', weights_only=False)
    dependencies = [Path(__file__), rp, fp, sp, jp, cp,
        Path('RECONSTRUCTION_CROSSFIT_NUMERIC_ABLATION_PROTOCOL_20260921.md')]
    bindings = {str(p): sha256(p) for p in dependencies}
    folder = root/'encoder_crossfit_numeric_ablation_20260921'
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    frames = []; hashes = {}; maximum_error = 0.
    for f in range(5):
        path = root/'encoder_crossfit_outer_20260921'/f'fold{f}.parquet'
        if sha256(path) != parent['prediction_hashes'][str(f)]:
            raise ValueError('Changed prediction file')
        dest = folder/f'fold{f}.parquet'; meta = folder/f'fold{f}.json'
        if dest.exists() and meta.exists():
            info = json.loads(meta.read_text())
            if info['bindings'] != bindings or sha256(dest) != info['sha256']:
                raise ValueError('Changed diagnostic cache')
            frames.append(pd.read_parquet(dest)); hashes[str(f)] = info['sha256']
            maximum_error = max(maximum_error, info['maximum_reproduction_error'])
            continue
        pred = pd.read_parquet(path)
        pred = pred.loc[pred.target.eq(target) & pred.status.eq('ok')].set_index('patient')
        ck = torch.load(root/'encoder_crossfit_training_20260920'/f'fold{f}_real.pt', weights_only=False)
        model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
        model.load_state_dict(ck['state_dict'], strict=True)
        rows = []; maximum_fold_error = 0.
        with torch.inference_mode():
            for shard in plan['shards']:
                person = shard['patient']
                if person not in pred.index:
                    continue
                if plan['folds'][f][person] != 'test':
                    raise ValueError('Not outer patient')
                p = Path(shard['path'])
                if sha256(p) != shard['sha256']:
                    raise ValueError('Changed shard')
                e = torch.load(p, weights_only=False)['samples'][col]
                row = pred.loc[person].to_dict(); row['patient'] = person
                if row['truth'] != e['truth_normalized']:
                    raise ValueError('Label mismatch')
                e = supported_example(e, scalers[f]); x = e['tokens']
                def predict(tokens):
                    value = float(model(tokens, e['event_hours'], e['anchor_hour'], target)[0])
                    return float(inverse_value(value, col, scalers[f]))
                original = predict(x)
                error = abs(original-row['real'])
                maximum_fold_error = max(maximum_fold_error, error)
                if not np.isfinite(original) or error > 1e-6:
                    raise ValueError('Original prediction not reproduced')
                for arm in INPUTS:
                    c = variables.index(arm); observed = bool(x[:, n+c].any())
                    value = predict(numeric_ablation(x, c)) if observed else original
                    if not np.isfinite(value):
                        raise ValueError('Nonfinite ablation')
                    row[arm] = value; row[f'{arm}_observed'] = observed
                rows.append(row)
        frame = pd.DataFrame(rows)
        if len(frame) != len(pred) or frame.patient.duplicated().any():
            raise ValueError('Patient coverage changed')
        temp = dest.with_suffix('.tmp'); frame.to_parquet(temp, index=False); temp.replace(dest)
        digest = sha256(dest)
        meta.write_text(json.dumps(dict(bindings=bindings, sha256=digest,
            maximum_reproduction_error=maximum_fold_error), indent=2)+'\n')
        frames.append(frame); hashes[str(f)] = digest
        maximum_error = max(maximum_error, maximum_fold_error)
        print('Completed fold', f, 'patients', len(frame), flush=True)
    full = pd.concat(frames, ignore_index=True)
    if full.patient.duplicated().any():
        raise ValueError('Duplicate patient across folds')
    expected = parent['targets'][target]['all']['eligible']
    if len(full) != expected:
        raise ValueError('Overall patient coverage changed')
    result = {}
    for arm in INPUTS:
        groups = {g: scores(p, arm) for g, p in full.groupby('group')}
        components = [dict(component_ordinal=i, groups={g: scores(q, arm) for g, q in p.groupby('group')})
                      for i, (_, p) in enumerate(full.groupby('component', sort=True))]
        result[arm] = dict(groups=groups, components=components)
    output.write_text(json.dumps(dict(target=target, patients=len(full), interventions=result,
        maximum_reproduction_error=maximum_error, bindings=bindings, prediction_hashes=hashes,
        runtime_modified=False, promotion_allowed=False,
        limitation='Posthoc development numerical sensitivity only. Median replacement may be out of distribution. No causal effect, feature selection, retraining or runtime promotion.'), indent=2)+'\n')
    print(json.dumps({a:r['groups'] for a,r in result.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()

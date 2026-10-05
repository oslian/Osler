"""Frozen all-target outer probe inference, with exact native reproduction."""

import argparse
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from scripts.audit_reconstruction_raw_state import summarize
from scripts.cache_nonlinear_probe import encode_features
from scripts.encoder_fold_scaling import inverse_value, transform_example
from scripts.score_additional_reconstruction import verify_frame
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def frozen_predictions(bundle, body, raw, target):
    if bundle['target'] != target or bundle['parameters'] != PARAMETERS:
        raise ValueError('Wrong frozen head identity')
    if set(bundle['models']) != {'body_probe', 'raw_probe', 'process_summary'}:
        raise ValueError('Incomplete fixed head family')
    if (body.shape != (len(raw), 64) or raw.shape != (len(body), 265)
            or not len(body) or not np.isfinite(body).all() or not np.isfinite(raw).all()):
        raise ValueError('Invalid fixed probe inputs')
    process = raw.copy()
    process[:, :106] = 0
    results = {}
    with threadpool_limits(limits=4):
        for arm, values in (('body_probe', body), ('raw_probe', raw), ('process_summary', process)):
            model = bundle['models'][arm]
            if model.n_iter_ != PARAMETERS['max_iter'] or any(model.get_params()[k] != v for k, v in PARAMETERS.items()):
                raise ValueError('Changed trained head settings')
            prediction = np.asarray(model.predict(values))
            if prediction.shape != (len(raw),) or not np.isfinite(prediction).all():
                raise ValueError('Nonfinite or incomplete frozen predictions')
            results[arm] = prediction
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    fold = parser.parse_args().fold
    root = Path('data/research_recovery/additional_development_domains_20260921')
    private = root / 'nonlinear_probe_20261001'
    dest = private / 'outer_20261002'
    dest.mkdir(exist_ok=True)
    output = dest / f'fold{fold}.parquet'
    report_path = Path(f'artifacts/native_reconstruction_nonlinear_outer_fold{fold}_20261002.json')
    if report_path.exists():
        raise ValueError('Existing completed outer inference')
    training_path = Path(f'artifacts/native_reconstruction_nonlinear_training_fold{fold}_20261001.json')
    training = json.loads(training_path.read_text())
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    source_report = json.loads(source_path.read_text())
    native_meta_path = root / f'continuation_outer_20260929/fold{fold}.json'
    native_meta = json.loads(native_meta_path.read_text())
    checked = {}
    for report in (training, source_report, native_meta):
        verify_bound_files(report['bindings'], checked)
    if training['fold'] != fold or training['parameters'] != PARAMETERS:
        raise ValueError('Wrong training report')
    fp, sp = root / 'partitions_20260922.pt', root / 'scaling_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds'][fold]
    scaling = torch.load(sp, weights_only=False)
    variables, scaler = scaling['variables'], scaling['scalers'][fold]
    base_path = root / f'coverage_outer_20260927/fold{fold}.parquet'
    native_path = root / f'continuation_outer_20260929/fold{fold}.parquet'
    base, native = pd.read_parquet(base_path), pd.read_parquet(native_path)
    verify_frame(base, scaling, roles, fold)
    pd.testing.assert_frame_equal(base[['patient', 'target', 'fold']], native[['patient', 'target', 'fold']])
    base = base.set_index(['patient', 'target'])
    native = native.set_index(['patient', 'target'])
    schema_path = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(schema_path, weights_only=False)
    if variables != schema['variables'] or len(variables) != 53:
        raise ValueError('Changed input schema')
    source_fold = 0 if fold == 2 else fold
    weight_path = root / f'monitored_continuation_20260929/fold{source_fold}_real_epoch10.pt'
    if sha256(weight_path) != native_meta['bindings'][str(weight_path)]:
        raise ValueError('Wrong frozen native checkpoint')
    checkpoint = torch.load(weight_path, weights_only=False)
    if (checkpoint['fold'], checkpoint['epoch'], checkpoint['arm']) != (source_fold, 10, 'real'):
        raise ValueError('Wrong native checkpoint identity')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    cache_path = root / 'full_histories_20260922.sqlite'
    history = ReconstructionCache(cache_path)
    keys = [(p, int(c)) for p, c in history.db.execute('SELECT patient,target FROM examples ORDER BY patient,target')
            if roles[p] == 'test']
    if {(p, variables[c]) for p, c in keys} != set(base.index):
        raise ValueError('Incomplete outer query grid')
    paths = [Path(__file__), Path('scripts/cache_nonlinear_probe.py'),
             Path('scripts/train_nonlinear_probe.py'), Path('scripts/audit_reconstruction_raw_state.py'),
             Path('scripts/score_additional_reconstruction.py'), Path('scripts/encoder_fold_scaling.py'),
             Path('scripts/train_encoder_crossfit.py'), Path('scripts/score_monitored_continuation_outer.py'),
             Path('RECONSTRUCTION_NONLINEAR_OUTER_PROTOCOL_20261002.md'),
             training_path, source_path, native_meta_path, fp, sp, base_path, native_path,
             schema_path, weight_path, cache_path]
    for target, meta in training['targets'].items():
        path = private / f'fold{fold}_probes/{target}.joblib'
        if sha256(path) != meta['model_sha256']:
            raise ValueError('Changed source-fitted probe')
        paths.append(path)
    bindings = {str(p): sha256(p) for p in paths}
    contract = dict(fold=fold, role='test', bindings=bindings)
    db_path = dest / f'fold{fold}_features.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS features (patient TEXT NOT NULL,target INTEGER NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(patient,target))')
    encoded = json.dumps(contract, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if old is not None and old[0] != encoded:
        raise ValueError('Changed outer extraction contract')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (encoded,))
    db.commit()
    existing = set(db.execute('SELECT patient,target FROM features'))
    if not existing <= set(keys):
        raise ValueError('Foreign outer cached query')
    captured = []
    hook = model.readout.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy().copy()[0]))
    try:
        with torch.inference_mode():
            for number, (patient, c) in enumerate(keys, 1):
                if (patient, c) in existing:
                    continue
                example = transform_example(history.get(patient, c), scaler)
                captured.clear()
                pred = float(model(example['tokens'], example['event_hours'], example['anchor_hour'], variables[c])[0])
                if len(captured) != 1:
                    raise ValueError('Invalid readout capture')
                raw, _ = summarize(example['tokens'], example['event_hours'], example['anchor_hour'], variables, variables[c])
                if not np.isclose(inverse_value(pred, c, scaler), native.loc[(patient, variables[c]), 'real10'], rtol=0, atol=1e-7):
                    raise ValueError('Changed original prediction')
                if not np.isclose(inverse_value(example['truth_normalized'], c, scaler), base.loc[(patient, variables[c]), 'truth'], rtol=1e-12, atol=1e-10):
                    raise ValueError('Changed original truth')
                payload, digest = encode_features(captured[0], raw, example['truth_normalized'], pred)
                db.execute('INSERT INTO features VALUES (?,?,?,?)', (patient, c, payload, digest))
                if number % 1000 == 0:
                    db.commit()
                if number % 20000 == 0:
                    print('Outer feature extraction', fold, number, len(keys), flush=True)
        db.commit()
        if set(db.execute('SELECT patient,target FROM features')) != set(keys):
            raise ValueError('Incomplete final outer grid')
        frames = []
        supported = set(base.index.get_level_values('target'))
        unsupported = sorted(set(training['targets']) - supported)
        for target in sorted(training['targets']):
            if target not in supported:
                continue
            c = variables.index(target)
            ids, features = load_target(db, c)
            bundle = joblib.load(private / f'fold{fold}_probes/{target}.joblib')
            predictions = frozen_predictions(bundle, features['body'], features['raw'], target)
            original = base.loc[[(p, target) for p in ids]]
            predictions['native'] = features['prediction']
            predictions['population'] = (original.population.to_numpy() - scaler['location'][c]) / scaler['scale'][c]
            if not np.allclose(features['truth'], (original.truth.to_numpy() - scaler['location'][c]) / scaler['scale'][c], rtol=1e-10, atol=1e-10):
                raise ValueError('Changed cached normalized truth')
            frames.append(pd.DataFrame(dict(patient=ids, component=original.component.to_numpy(),
                fold=fold, target=target, truth=features['truth'], **predictions)))
        full = pd.concat(frames, ignore_index=True)
        if len(full) != len(keys) or full.duplicated(['patient', 'target']).any():
            raise ValueError('Incomplete final outer predictions')
        tmp = output.with_suffix('.tmp')
        full.to_parquet(tmp, index=False)
        tmp.replace(output)
    finally:
        hook.remove()
        history.close()
        db.close()
    report_path.write_text(json.dumps(dict(contract, examples=len(full), patients=int(full.patient.nunique()),
        targets=int(full.target.nunique()), unsupported_targets=unsupported,
        feature_sha256=sha256(db_path), prediction_sha256=sha256(output),
        runtime_modified=False, promotion_allowed=False,
        limitation='Frozen full outer current-value reconstruction probe. No fitting or selection. Reused development, not new external validation or forecast.'), indent=2) + '\n')
    print('Frozen full outer inference complete', fold, len(full), flush=True)


if __name__ == '__main__':
    main()

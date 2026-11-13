"""Cache complete masked body/raw features for a fixed source-only probe."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch

from scripts.audit_reconstruction_raw_state import summarize
from scripts.encoder_fold_scaling import inverse_value, transform_example
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def encode_features(body, raw, truth, prediction):
    body, raw = np.asarray(body, dtype=np.float32), np.asarray(raw, dtype=np.float64)
    if body.shape != (64,) or raw.shape != (265,) or not np.isfinite(np.r_[body, raw, truth, prediction]).all():
        raise ValueError('Invalid probe feature contract')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, body=body, raw=raw, truth=np.float64(truth), prediction=np.float64(prediction))
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def decode_features(payload, digest):
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Corrupt feature payload')
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        result = {name: values[name].copy() for name in ('body', 'raw', 'truth', 'prediction')}
    encode_features(**result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--role', choices=('fit', 'selection'), required=True)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    folder = root/'nonlinear_probe_20261001'
    folder.mkdir(exist_ok=True)
    path = folder/f'fold{args.fold}_{args.role}.sqlite'
    report_path = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{args.fold}_{args.role}_20261001.json')
    if report_path.exists():
        raise ValueError('Existing completed feature artifact')
    proof_path = Path(f'artifacts/native_reconstruction_continuation_selection_fold{args.fold}_verification_20260930.json')
    proof = json.loads(proof_path.read_text())
    verify_bound_files(proof['bindings'], {})
    meta_path = root/f'continuation_outer_20260929/fold{args.fold}.json'
    meta = json.loads(meta_path.read_text())
    verify_bound_files(meta['bindings'], {})
    fp, sp = root/'partitions_20260922.pt', root/'scaling_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds'][args.fold]
    scaling = torch.load(sp, weights_only=False)
    variables, scaler = scaling['variables'], scaling['scalers'][args.fold]
    schema_path = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(schema_path, weights_only=False)
    if variables != schema['variables'] or len(variables) != 53:
        raise ValueError('Changed feature schema')
    source_fold = 0 if args.fold == 2 else args.fold
    weight_path = root/f'monitored_continuation_20260929/fold{source_fold}_real_epoch10.pt'
    if sha256(weight_path) != meta['bindings'][str(weight_path)]:
        raise ValueError('Changed frozen weights')
    checkpoint = torch.load(weight_path, weights_only=False)
    if (checkpoint['fold'], checkpoint['epoch'], checkpoint['arm']) != (source_fold, 10, 'real'):
        raise ValueError('Wrong frozen model')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    cache_path = root/'full_histories_20260922.sqlite'
    source = ReconstructionCache(cache_path)
    keys = [(p, int(c)) for p, c in source.db.execute('SELECT patient,target FROM examples ORDER BY patient,target')
            if roles[p] == args.role]
    expected = {(p, int(c)) for i, p in enumerate(scaling['people']) if roles[p] == args.role
                for c in np.flatnonzero(np.isfinite(scaling['centers'][i]))}
    if set(keys) != expected:
        raise ValueError('Incomplete eligible query grid')
    selection_path = root/f'continuation_selection_20260930/fold{args.fold}.parquet'
    baseline_path = root/f'coverage_selection_20260926/fold{args.fold}.parquet'
    selection = pd.read_parquet(selection_path).set_index(['patient', 'target'])
    baseline = pd.read_parquet(baseline_path).set_index(['patient', 'target'])
    if args.role == 'selection' and set((p, variables[c]) for p, c in keys) != set(selection.index):
        raise ValueError('Changed selection population')
    paths = [Path(__file__), proof_path, meta_path, fp, sp, schema_path, weight_path,
             cache_path, selection_path, baseline_path,
             Path('RECONSTRUCTION_NONLINEAR_PROBE_PROTOCOL_20261001.md'),
             Path('scripts/audit_reconstruction_raw_state.py'), Path('scripts/encoder_fold_scaling.py'),
             Path('scripts/train_encoder_crossfit.py'), Path('scripts/score_monitored_continuation_outer.py')]
    bindings = {str(p): sha256(p) for p in paths}
    contract = dict(fold=args.fold, source_fold=source_fold, role=args.role, bindings=bindings,
                    body_width=64, raw_width=265)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS features (patient TEXT NOT NULL,target INTEGER NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,PRIMARY KEY(patient,target))')
    encoded = json.dumps(contract, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if old is not None and old[0] != encoded:
        raise ValueError('Changed feature extraction contract')
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (encoded,))
    db.commit()
    existing = set(db.execute('SELECT patient,target FROM features'))
    if not existing <= expected:
        raise ValueError('Foreign cached query')
    captured = []
    hook = model.readout.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu().numpy().copy()[0]))
    try:
        with torch.inference_mode():
            for number, (patient, c) in enumerate(keys, 1):
                if (patient, c) in existing:
                    continue
                example = transform_example(source.get(patient, c), scaler)
                captured.clear()
                pred = float(model(example['tokens'], example['event_hours'], example['anchor_hour'], variables[c])[0])
                if len(captured) != 1:
                    raise ValueError('Missing or repeated body capture')
                raw, _ = summarize(example['tokens'], example['event_hours'], example['anchor_hour'], variables, variables[c])
                value = float(inverse_value(pred, c, scaler))
                if args.role == 'selection':
                    key = (patient, variables[c])
                    if not np.isclose(value, selection.loc[key, 'real10'], atol=1e-7, rtol=0):
                        raise ValueError('Original frozen prediction changed')
                    truth = inverse_value(example['truth_normalized'], c, scaler)
                    if not np.isclose(truth, baseline.loc[key, 'truth'], atol=1e-10, rtol=1e-12):
                        raise ValueError('Changed selection truth')
                payload, digest = encode_features(captured[0], raw, example['truth_normalized'], pred)
                db.execute('INSERT INTO features VALUES (?,?,?,?)', (patient, c, payload, digest))
                if number % 1000 == 0:
                    db.commit()
                if number % 20000 == 0:
                    print('Cached', args.fold, args.role, number, len(keys), flush=True)
        db.commit()
        if set(db.execute('SELECT patient,target FROM features')) != expected:
            raise ValueError('Incomplete final feature grid')
        # Recheck resumed payloads too; a partial file is not completion evidence.
        for payload, digest in db.execute('SELECT payload,digest FROM features'):
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError('Corrupt final feature payload')
    finally:
        hook.remove()
        source.close()
        db.close()
    report = dict(contract, examples=len(keys), patients=len({p for p, _ in keys}),
                  targets=len({c for _, c in keys}), private_sha256=sha256(path),
                  runtime_modified=False, promotion_allowed=False,
                  limitation='Feature extraction only, not trained probe or predictive efficacy. Current masked reconstruction, reused source roles, frozen epoch10 encoder. Local private payloads include identities; never publish the cache.')
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'bindings'}, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Resumable full-grid organ extraction with native body/raw/point parity."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import torch

from scripts.audit_reconstruction_raw_state import summarize
from scripts.cache_nonlinear_probe import decode_features
from scripts.encoder_fold_scaling import transform_example
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache

ROOT = Path('data/research_recovery/additional_development_domains_20260921')
DEST = ROOT/'full_organ_compartments_20261003'
PROTOCOL = Path('RECONSTRUCTION_FULL_ORGAN_PROTOCOL_20261003.md')
RESERVE = 10 * 1024**3


def payload_digest(patient, target, reference_digest, payload):
    identity = json.dumps([str(patient), int(target), reference_digest], separators=(',', ':')).encode()
    return hashlib.sha256(identity+b'\0'+payload).hexdigest()


def pack_organs(organs):
    organs = np.asarray(organs)
    if organs.ndim != 2 or not all(organs.shape) or organs.dtype != np.float32 or not np.isfinite(organs).all():
        raise ValueError('Finite two-dimensional float32 organ state required')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, organs=organs)
    return buffer.getvalue()


def unpack_organs(patient, target, reference_digest, payload, digest, shape):
    if payload_digest(patient, target, reference_digest, payload) != digest:
        raise ValueError('Corrupt or misidentified organ payload')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if data.files != ['organs']:
            raise ValueError('Unexpected organ payload fields')
        result = data['organs'].copy()
    if result.shape != tuple(shape):
        raise ValueError('Changed organ dimensions')
    pack_organs(result)
    return result


def capture_native(model, example, target):
    if model.training:
        raise ValueError('Frozen evaluation model required')
    captured = []

    def capture(module, inputs, output):
        state = output['state']
        captured.append((state.body[0].detach().cpu().numpy().copy(),
                         state.organs[0].detach().cpu().numpy().copy()))

    hook = model.trajectory.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            prediction = float(model(example['tokens'], example['event_hours'], example['anchor_hour'], target)[0])
        if len(captured) != 1:
            raise ValueError('Expected exactly one native trajectory state')
        body, organs = captured[0]
        if not np.isfinite(body).all() or not np.isfinite(prediction):
            raise ValueError('Nonfinite native output')
        pack_organs(organs)
        return body, organs, prediction
    finally:
        hook.remove()


def checked_organs(model, example, target, reference):
    if float(example['truth_normalized']) != float(reference['truth']):
        raise ValueError('Changed normalized truth')
    raw, _ = summarize(example['tokens'], example['event_hours'], example['anchor_hour'],
                       model.trajectory.variables, target)
    np.testing.assert_array_equal(raw, reference['raw'])
    body, organs, prediction = capture_native(model, example, target)
    np.testing.assert_allclose(body, reference['body'], atol=1e-6, rtol=0)
    np.testing.assert_allclose(prediction, reference['prediction'], atol=1e-7, rtol=0)
    return organs, float(np.max(np.abs(body-reference['body']))), abs(prediction-float(reference['prediction']))


def open_cache(path, reference_path, contract, roles):
    db = sqlite3.connect(path, uri=True)
    try:
        db.execute('ATTACH DATABASE ? AS reference', (reference_path.resolve().as_uri()+'?mode=ro',))
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS compartments (patient TEXT NOT NULL,target INTEGER NOT NULL,'
                   'reference_digest TEXT NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,'
                   'body_error REAL NOT NULL,point_error REAL NOT NULL,PRIMARY KEY(patient,target))')
        encoded = json.dumps(contract, sort_keys=True)
        prior = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if prior is not None and prior[0] != encoded:
            raise ValueError('Changed full organ extraction contract')
        for patient, in db.execute('SELECT DISTINCT patient FROM reference.features'):
            if roles.get(patient) != contract['role']:
                raise ValueError('Foreign or wrong-role reference patient')
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (encoded,))
        db.commit()
        validate_cache(db, contract['organ_shape'])
        return db
    except BaseException:
        db.close()
        raise


def validate_cache(db, shape):
    foreign = db.execute('SELECT COUNT(*) FROM compartments c LEFT JOIN reference.features r '
        'ON c.patient=r.patient AND c.target=r.target WHERE r.patient IS NULL OR c.reference_digest!=r.digest').fetchone()[0]
    if foreign:
        raise ValueError('Foreign or changed reference query')
    for row in db.execute('SELECT patient,target,reference_digest,payload,digest,body_error,point_error FROM compartments'):
        unpack_organs(*row[:5], shape)
        if not (np.isfinite(row[5:]).all() and 0 <= row[5] <= 1e-6 and 0 <= row[6] <= 1e-7):
            raise ValueError('Invalid cached native parity')
    total = db.execute('SELECT COUNT(*) FROM reference.features').fetchone()[0]
    stored = db.execute('SELECT COUNT(*) FROM compartments').fetchone()[0]
    if total == 0:
        raise ValueError('Empty reference grid')
    return dict(expected=total, stored=stored, remaining=total-stored, complete=stored == total)


def append_batch(db, shape, extract, max_new, space_check):
    if max_new < 1:
        raise ValueError('A positive finite batch budget is required')
    query = ('SELECT r.patient,r.target FROM reference.features r '
             'LEFT JOIN compartments c ON r.patient=c.patient AND r.target=c.target '
             'WHERE c.patient IS NULL ORDER BY r.patient,r.target LIMIT ?')
    # The finite key list avoids changing the table driving a live join cursor.
    keys = list(db.execute(query, (max_new,)))
    added = 0
    try:
        for patient, target in keys:
            if added % 128 == 0:
                space_check()
            payload, digest = db.execute('SELECT payload,digest FROM reference.features WHERE patient=? AND target=?',
                                        (patient, target)).fetchone()
            reference = decode_features(payload, digest)
            organs, body_error, point_error = extract(patient, target, reference)
            if np.asarray(organs).shape != tuple(shape):
                raise ValueError('Unexpected extracted organ dimensions')
            if not (np.isfinite([body_error, point_error]).all() and 0 <= body_error <= 1e-6 and 0 <= point_error <= 1e-7):
                raise ValueError('Native parity failed')
            payload = pack_organs(organs)
            db.execute('INSERT INTO compartments VALUES (?,?,?,?,?,?,?)',
                (patient, target, digest, payload, payload_digest(patient, target, digest, payload), body_error, point_error))
            added += 1
            if added % 128 == 0:
                db.commit()
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return added, validate_cache(db, shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--role', choices=('fit', 'selection'), required=True)
    parser.add_argument('--max-new', type=int, required=True)
    args = parser.parse_args()
    if args.max_new < 1:
        parser.error('--max-new must be positive')
    start = time.monotonic()
    reference = ROOT/f'nonlinear_probe_20261001/fold{args.fold}_{args.role}.sqlite'
    rp = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{args.fold}_{args.role}_20261001.json')
    report = json.loads(rp.read_text())
    if (report['fold'], report['role']) != (args.fold, args.role):
        raise ValueError('Wrong reference role/fold')
    checked = {}
    verify_bound_files(report['bindings'], checked)
    if sha256(reference) != report['private_sha256']:
        raise ValueError('Changed full reference cache')
    fp, sp, hp = [ROOT/n for n in ('partitions_20260922.pt', 'scaling_20260922.pt', 'full_histories_20260922.sqlite')]
    roles = torch.load(fp, weights_only=False)['folds'][args.fold]
    scaling = torch.load(sp, weights_only=False)
    variables, scaler = scaling['variables'], scaling['scalers'][args.fold]
    schema_path = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(schema_path, weights_only=False)
    if variables != schema['variables'] or len(variables) != 53:
        raise ValueError('Changed canonical variable schema')
    sf = 0 if args.fold == 2 else args.fold
    wp = ROOT/f'monitored_continuation_20260929/fold{sf}_real_epoch10.pt'
    weights = torch.load(wp, weights_only=False)
    if (weights['fold'], weights['epoch'], weights['arm']) != (sf, 10, 'real'):
        raise ValueError('Wrong frozen weights')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    model.load_state_dict(weights['state_dict'], strict=True)
    layer = model.trajectory.layer
    shape = (layer.organ_count, layer.organ_dim)
    sources = [Path(__file__), PROTOCOL, rp, reference, fp, sp, hp, wp, schema_path,
        Path('scripts/cache_nonlinear_probe.py'), Path('scripts/encoder_fold_scaling.py'),
        Path('scripts/audit_reconstruction_raw_state.py'), Path('scripts/train_encoder_crossfit.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'), Path('whole_body_jepa/native_organ_state_residual.py'),
        Path('whole_body_jepa/native_organ_observations.py'), Path('whole_body_jepa/physiology_state.py'),
        Path('whole_body_jepa/reconstruction_cache.py'), Path('whole_body_joint_jepa.py')]
    bindings = {str(p): sha256(p) for p in sources}
    contract = dict(fold=args.fold, source_fold=sf, role=args.role, variables=variables,
        organ_names=list(model.trajectory.organ_names), organ_shape=list(shape), bindings=bindings)
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST/f'fold{args.fold}_{args.role}.sqlite'
    db = open_cache(path, reference, contract, roles)
    source = ReconstructionCache(hp)
    try:
        def extract(patient, target, ref):
            example = transform_example(source.get(patient, target), scaler)
            return checked_organs(model, example, variables[target], ref)

        def space_check():
            if shutil.disk_usage(DEST).free < RESERVE:
                raise RuntimeError('Organ extraction disk reserve reached; committed rows retained')

        added, status = append_batch(db, shape, extract, args.max_new, space_check)
        if status['expected'] != report['examples']:
            raise ValueError('Reference count differs from completed source report')
        errors = db.execute('SELECT MAX(body_error),MAX(point_error),SUM(LENGTH(payload)) FROM compartments').fetchone()
        counts = dict(db.execute('SELECT target,COUNT(*) FROM compartments GROUP BY target'))
        supports = {variables[c]: dict(expected=n, stored=counts.get(c, 0))
                    for c, n in db.execute('SELECT target,COUNT(*) FROM reference.features GROUP BY target')}
    finally:
        source.close()
        db.close()
    result = dict(contract=contract, **status, added=added, max_new=args.max_new,
        wall_seconds=time.monotonic()-start, private_bytes=path.stat().st_size,
        private_sha256=sha256(path), organ_payload_bytes=errors[2] or 0,
        body_max_error=errors[0], point_max_error=errors[1], support=supports,
        runtime_modified=False, promotion_allowed=False,
        limitation='Extraction/parity only, not accuracy. Partial batches do not complete the full-grid experiment.')
    output = Path(f'artifacts/native_reconstruction_full_organs_fold{args.fold}_{args.role}_20261003.json')
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('contract', 'support', 'private_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()

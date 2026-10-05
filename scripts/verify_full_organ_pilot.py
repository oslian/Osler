"""Independent stepwise replay of the admitted full-grid extraction pilot."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import torch

from scripts.cache_full_organ_compartments import DEST, ROOT, unpack_organs
from scripts.cache_nonlinear_probe import decode_features
from scripts.encoder_fold_scaling import transform_example
from scripts.probe_organ_compartments import replay_full
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    output = Path('artifacts/native_reconstruction_full_organs_pilot_verification_20261003.json')
    if output.exists():
        raise ValueError('Completed pilot verification already exists')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    checked, bindings, results = {}, {}, {}
    for path in (Path(__file__), Path('scripts/probe_organ_compartments.py')):
        bindings[str(path)] = sha256(path)
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    try:
        for role in ('fit', 'selection'):
            rp = Path(f'artifacts/native_reconstruction_full_organs_fold0_{role}_20261003.json')
            report = json.loads(rp.read_text())
            contract = report['contract']
            if (contract['fold'], contract['role'], report['stored'], report['complete']) != (0, role, 512, False):
                raise ValueError('Not the prespecified engineering pilot')
            verify_bound_files(contract['bindings'], checked)
            bindings.update(contract['bindings'])
            path = DEST/f'fold0_{role}.sqlite'
            if sha256(path) != report['private_sha256']:
                raise ValueError('Pilot snapshot changed')
            scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)
            variables, scaler = scaling['variables'], scaling['scalers'][0]
            schema = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
            checkpoint = torch.load(ROOT/'monitored_continuation_20260929/fold0_real_epoch10.pt', weights_only=False)
            model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            ref = ROOT/f'nonlinear_probe_20261001/fold0_{role}.sqlite'
            count = 0
            commitment = hashlib.sha256()
            errors = dict(body=0., organs=0., native=0.)
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db, \
                 closing(sqlite3.connect(ref.resolve().as_uri()+'?mode=ro', uri=True)) as reference, torch.no_grad():
                rows = list(db.execute('SELECT patient,target,reference_digest,payload,digest FROM compartments ORDER BY patient,target'))
                keys = list(reference.execute('SELECT patient,target FROM features ORDER BY patient,target LIMIT 512'))
                if [(r[0], r[1]) for r in rows] != keys:
                    raise ValueError('Pilot grid changed')
                for patient, target, ref_digest, payload, digest in rows:
                    organs = unpack_organs(patient, target, ref_digest, payload, digest, contract['organ_shape'])
                    ref_payload, rd = reference.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?',
                                                       (patient, target)).fetchone()
                    if rd != ref_digest:
                        raise ValueError('Reference identity changed')
                    old = decode_features(ref_payload, rd)
                    e = transform_example(source.get(patient, target), scaler)
                    clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, variables[target])
                    body, proposals, presence = encode_native_organ_observations(model.trajectory.jepa, variables, clean)
                    state = replay_full(model.trajectory.layer, dict(body=body, organs=proposals, presence=presence,
                        times=times, anchor=e['anchor_hour']))
                    point = float(model.readout(state.body)[0, target])
                    np.testing.assert_array_equal(state.body[0].numpy(), old['body'])
                    np.testing.assert_array_equal(state.organs[0].numpy(), organs)
                    if point != float(old['prediction']) or float(e['truth_normalized']) != float(old['truth']):
                        raise ValueError('Native prediction/truth changed')
                    for k, err in (('body', np.max(abs(state.body[0].numpy()-old['body']))),
                                   ('organs', np.max(abs(state.organs[0].numpy()-organs))),
                                   ('native', abs(point-float(old['prediction'])))):
                        errors[k] = max(errors[k], float(err))
                    commitment.update(digest.encode()+b'\n')
                    count += 1
            results[role] = dict(verified=count, errors=errors, prefix_digest=commitment.hexdigest(),
                extraction_snapshot=report,
                note='Snapshot only; future appended rows require their own checks. Prefix digest binds these ordered rows.')
            print('Independent organ replay verified', role, count, errors, flush=True)
    finally:
        source.close()
    output.write_text(json.dumps(dict(results=results, bindings=bindings, runtime_modified=False,
        promotion_allowed=False, limitation='1024 extraction checks, not full E25 completion or model efficacy.'),
        indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

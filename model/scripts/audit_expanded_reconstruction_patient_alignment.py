"""Frozen prediction alignment diagnostic, not a new promotion test."""
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def alignment(truth, prediction):
    truth=np.asarray(truth,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if truth.ndim!=1 or prediction.shape!=truth.shape or len(truth)<2:
        raise ValueError('at least two aligned patients required')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('finite values required')
    own=np.abs(truth-prediction)
    cross=np.abs(truth[:,None]-prediction[None,:])
    # Exact expected error when each recipient gets a uniformly chosen OTHER
    # patient's prediction, conditional on this fixed role/target cohort.
    wrong=(cross.sum(axis=1)-own)/(len(truth)-1)
    return dict(original_mae=float(own.mean()),expected_wrong_patient_mae=float(wrong.mean()),
                alignment_benefit=float((wrong-own).mean()),prediction_std=float(prediction.std()),
                expected_absolute_prediction_shift=float((np.abs(prediction[:,None]-prediction[None,:]).sum())/(len(truth)*(len(truth)-1))))


def main():
    source=Path('artifacts/native_reconstruction_expanded_evaluation_20260913.json')
    report=json.loads(source.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed evaluation dependency')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
    saved=torch.load(root/'expanded_training_20260913/heldout_predictions.pt',weights_only=False)
    truth={r:{} for r in report['roles']}
    for shard in index['shards']:
        if shard['role'] not in truth:continue
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('changed sample')
        s=torch.load(shard['path'],weights_only=False)
        for c,e in s['samples'].items():truth[s['role']][(s['patient'],c)]=e['truth_normalized']
    results={}
    for role,targets in report['roles'].items():
        results[role]={}
        for target,row in targets.items():
            c=index['variables'].index(target)
            keys=[k for k in saved['predictions']['real'][role] if k[1]==c]
            if len(keys)<2:continue
            y=[truth[role][k] for k in keys]
            result={arm:alignment(y,[saved['predictions'][arm][role][k] for k in keys]) for arm in ('real','placebo')}
            for arm in result:
                if not np.isclose(result[arm]['original_mae'],row['normalized_mae'][arm],rtol=0,atol=1e-7):
                    raise ValueError('parent predictions fail reproduction')
            results[role][target]=dict(patients=len(keys),arms=result)
    output=Path('artifacts/native_reconstruction_expanded_alignment_20260914.json')
    if output.exists():raise ValueError('existing evidence')
    output.write_text(json.dumps(dict(complete=True,roles=results,runtime_modified=False,promotion_allowed=False,
        limitation='Conditional fixed-cohort wrong-patient diagnostic; no independent-data inference or physiology-vs-workflow separation.',
        bindings={str(p):sha256(p) for p in (Path(__file__),source)}),indent=2)+'\n')
    for role,rows in results.items():
        print(role,'positive alignment',sum(r['arms']['real']['alignment_benefit']>0 for r in rows.values()),'/',len(rows))


if __name__=='__main__':main()

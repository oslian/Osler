"""Reproduce all saved E21 predictions without fitting or changing candidates."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.train_content_retention import build_fold, predict_rows, shuffle_observed, ARMS
from whole_body_jepa.multisource_intervals import sha256


def verify_predictions(frame, prediction, arm):
    identity = ['patient', 'component', 'fold', 'context', 'variable', 'task', 'truth', 'frozen']
    pd.testing.assert_frame_equal(frame[identity], prediction[identity], check_exact=True)
    if not np.isfinite(prediction.prediction).all(): raise ValueError('Nonfinite saved prediction')
    np.testing.assert_allclose(frame[arm], prediction.prediction, atol=0, rtol=0)
    return len(frame)


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    path = Path('artifacts/native_reconstruction_content_training_assessment_20261003.json')
    assessment = json.loads(path.read_text())
    out = Path('artifacts/native_reconstruction_content_training_verification_20261003.json')
    if out.exists(): raise ValueError('Existing E21 verification')
    results = []
    for fold in range(5):
        report = Path(f'artifacts/native_reconstruction_content_training_fold{fold}_20261003.json')
        if assessment['bindings'][str(report)] != sha256(report): raise ValueError('Changed fold report')
        info = json.loads(report.read_text())
        variables, examples, original, contract, extraction, dest = build_fold(fold)
        if info['contract'] != contract or info['extraction'] != extraction:
            raise ValueError('Changed extraction')
        if {e['patient'] for e in examples['fit']} & {e['patient'] for e in examples['selection']}:
            raise ValueError('Patient overlap')
        private = dest/'predictions.parquet'
        if info['prediction_sha256'] != sha256(private): raise ValueError('Changed private predictions')
        frame = pd.read_parquet(private)
        _, _, eligible = shuffle_observed(torch.stack([e['values'] for e in examples['fit']]),
            torch.stack([e['mask'] for e in examples['fit']]))
        if info['eligible_variables'] != [v for v, ok in zip(variables, eligible) if ok]:
            raise ValueError('Changed eligible support')
        arms = {}
        for arm in ARMS:
            model_path = dest/f'{arm}.pt'
            if info['fits'][arm]['sha256'] != sha256(model_path): raise ValueError('Changed trained model')
            saved = torch.load(model_path, weights_only=False)
            model = deepcopy(original).eval(); model.load_state_dict(saved['model'], strict=True)
            head = nn.Linear(model.trajectory.layer.body_dim, len(variables)).eval()
            head.load_state_dict(saved['head'], strict=True)
            if not torch.equal(saved['eligible'], eligible): raise ValueError('Changed arm support')
            for key, value in original.trajectory.jepa.state_dict().items():
                if not torch.equal(value, model.trajectory.jepa.state_dict()[key]):
                    raise ValueError('Encoder not frozen')
            prediction = predict_rows(model, head, examples['selection'], eligible, fold, variables)
            arms[arm] = dict(predictions_reproduced=verify_predictions(frame, prediction, arm),
                             encoder_unchanged=True, max_absolute_error=0.)
        results.append(dict(fold=fold, contexts=len(examples['selection']), arms=arms))
        print('Verified E21', fold, len(frame), flush=True)
    out.write_text(json.dumps(dict(folds=results, assessment_sha256=sha256(path),
        verifier_sha256=sha256(Path(__file__)), runtime_modified=False, promotion_allowed=False),
        indent=2, allow_nan=False)+'\n')
    print('All 15 saved E21 models reproduced exactly; frozen encoders unchanged.', flush=True)


if __name__ == '__main__': main()

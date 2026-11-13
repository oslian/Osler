"""Matched source-only test of additional measured future supervision."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_event_aware_patient_state import patient_partition, paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support
from whole_body_jepa.multisource_intervals import sha256, SEEDS
from whole_body_jepa.runtime import load_release
from whole_body_joint_jepa import _masked_latent_prediction_loss


def shuffled_deltas(deltas, valid, subjects, fit):
    """Preserve each target's fit delta distribution without same-patient donors."""
    result = deltas.copy()
    for col in range(deltas.shape[1]):
        rows = np.flatnonzero(valid[:, col] & fit)
        if not len(rows):
            continue
        ordered = rows[np.argsort(subjects[rows], kind='stable')]
        _, counts = np.unique(subjects[ordered], return_counts=True)
        shift = int(counts.max())
        if 2*shift > len(rows):
            raise ValueError('a single patient dominates placebo labels')
        donors = np.roll(ordered, shift)
        if np.any(subjects[ordered] == subjects[donors]):
            raise ValueError('same patient placebo donor')
        result[ordered, col] = deltas[donors, col]
    return result


def auxiliary_loss(model, output, current, truth, valid, indices):
    losses = []
    for index in indices:
        active = valid[:, index]
        if not bool(active.any()):
            continue
        teacher = teacher_target(model, current[active], truth[active, index], index)
        value = F.smooth_l1_loss(output['value'][active, index], truth[active, index])
        latent = _masked_latent_prediction_loss(
            output['predicted_target_latent'][active, index], teacher,
            torch.ones(int(active.sum()), dtype=torch.bool))
        losses.append(value + latent)
    return torch.stack(losses).mean() if losses else output['value'].sum()*0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    seeds = tuple(SEEDS) if args.extended else (7,19,31)
    epochs = 64 if args.extended else 8
    stage = 'budget' if args.extended else 'pilot'
    protocol = Path('AUXILIARY_ENCODER_BUDGET_PROTOCOL_20260910.md' if args.extended
                    else 'AUXILIARY_ENCODER_PILOT_PROTOCOL_20260910.md')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report_path = Path(f'artifacts/encoder_auxiliary_{stage}_20260910.json')
    weights = Path(f'/private/tmp/encoder_auxiliary_{stage}_weights_20260910')
    if report_path.exists() or weights.exists():
        raise ValueError('use fresh outputs')
    support = json.loads(Path('artifacts/auxiliary_supervision_20260910.json').read_text())
    label_path = Path('/private/tmp/auxiliary_supervision_20260910.npz')
    if sha256(label_path) != support['binding'][str(label_path)]:
        raise ValueError('label artifact changed')
    label = np.load(label_path)
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    if list(label['variables']) != variables:
        raise ValueError('variable order differs')
    index = variables.index('glucose')
    aux = [variables.index(v) for v in support['eligible_auxiliary_variables']]
    frame = pd.DataFrame({'stay_id':label['stay_id'], 'subject_id':label['subject_id'],
                          't':label['query_time'], 'truth':label['future'][:, index]})
    original = pd.read_parquet('/private/tmp/availability_glucose_development.parquet')
    old = frame.subject_id.isin(set(original.subject_id)).to_numpy()
    raw_paths = [Path('/private/tmp/availability_jepa_eicu_labs.parquet'),
                 Path('/private/tmp/encoder_source_expansion_20260910/labs.parquet')]
    for path in raw_paths:
        if sha256(path) != support['binding'][str(path)]:
            raise ValueError('raw events changed')
    raw = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
    raw = raw.loc[raw.stay_id.isin(frame.stay_id)]
    base_arrays, rows = build_arrays(frame, raw, assets,
        {s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()}, truthful=True)
    medians, scales = np.asarray(assets.scaler.medians), np.asarray(assets.scaler.scales)
    normalized = (label['future']-medians)/scales
    weights.mkdir()
    report = dict(status=f'source_{stage}_only', runtime_modified=False, external_data_read=False,
        promotion_allowed=False, pretraining_patient_overlap_not_excluded=True,
        settings=dict(seeds=list(seeds), epochs=epochs, steps=16, batch_size=32, auxiliary_weight=.25,
                      selection='source tune glucose MAE including epoch zero',
                      auxiliary_targets=support['eligible_auxiliary_variables']), seeds=[],
        binding={str(p):sha256(p) for p in [Path(__file__), label_path, protocol,
                Path('models/whole_body_joint_jepa.pt'),Path('whole_body_joint_jepa.py'),
                Path('whole_body_jepa/availability_inputs.py')]})
    for seed in seeds:
        old_fit, old_tune = patient_partition(frame.loc[old, 'subject_id'], seed)
        fit = np.ones(len(frame), bool); fit[old] = old_fit
        tune = np.zeros(len(frame), bool); tune[old] = old_tune
        if frame.loc[fit, 'subject_id'].nunique() != 2250 or frame.loc[tune,'subject_id'].nunique() !=136:
            raise ValueError('cohort mismatch')
        allowed = source_patient_support(base_arrays, rows, frame.subject_id, fit)>0
        arrays = restrict_input_support(base_arrays, allowed, assets.model.variable_module_ids.numpy())
        valid = label['paired'] & allowed[None, :]
        if not np.all(arrays['mask'][rows][valid]>0):
            raise ValueError('paired anchor missing from encoder')
        residuals = normalized-arrays['current'][rows]
        auxiliary_valid = valid.copy()
        auxiliary_valid[:, [i for i in range(len(variables)) if i not in aux]] = False
        placebo = arrays['current'][rows]+shuffled_deltas(residuals, auxiliary_valid, frame.subject_id.to_numpy(), fit)
        pools = [np.flatnonzero(fit & frame.subject_id.eq(s).to_numpy())
                 for s in sorted(set(frame.loc[fit,'subject_id']))]
        result = dict(seed=seed, arms={}, comparisons={})
        predictions = {}
        for arm in ('glucose_only','auxiliary','shuffled_auxiliary'):
            torch.manual_seed(seed)
            model = deepcopy(assets.model)
            model.value_delta_activation = torch.nn.Identity()
            model.belief_input_mode = 'observed_steps'
            params = configure(model, True)
            optimizer = torch.optim.AdamW(params, lr=.0001, weight_decay=.0001)
            initial = predict(model, arrays, rows[tune], index, medians[index], scales[index])
            best_mae = patient_equal_mae(frame.loc[tune], frame.loc[tune,'truth'], initial)
            best, best_epoch, log = deepcopy(model.state_dict()), 0, []
            rng = np.random.default_rng(seed)
            for epoch in range(epochs):
                for _ in range(16):
                    selected = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    inputs = batch(arrays, rows[selected])
                    output = model(*inputs)
                    truth = torch.tensor(normalized[selected], dtype=torch.float32)
                    teacher = teacher_target(model, inputs[1], truth[:, index], index)
                    loss = F.smooth_l1_loss(output['value'][:,index], truth[:,index])
                    loss = loss + _masked_latent_prediction_loss(
                        output['predicted_target_latent'][:,index], teacher, torch.ones(len(selected),dtype=torch.bool))
                    if arm != 'glucose_only':
                        auxiliary_truth = torch.tensor(placebo[selected],dtype=torch.float32) if arm=='shuffled_auxiliary' else truth
                        loss = loss + .25*auxiliary_loss(model,output,inputs[1],auxiliary_truth,
                                                       torch.from_numpy(valid[selected]),aux)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,1.)
                    optimizer.step(); model.update_target_encoder()
                pred = predict(model,arrays,rows[tune],index,medians[index],scales[index])
                mae = patient_equal_mae(frame.loc[tune],frame.loc[tune,'truth'],pred)
                if mae < best_mae:
                    best_mae,best_epoch,best = mae,epoch+1,deepcopy(model.state_dict())
                log.append(dict(epoch=epoch+1,tune_mae=mae))
                print(json.dumps(dict(seed=seed,arm=arm,**log[-1])),flush=True)
            model.load_state_dict(best)
            pred=predict(model,arrays,rows[tune],index,medians[index],scales[index])
            predictions[arm]=pred
            np.save(weights/f'{arm}_{seed}_tune.npy',pred)
            torch.save(dict(state_dict=best, value_delta_activation='identity',belief_input_mode='observed_steps',
                allowed_variables=allowed.tolist(),variables=variables,status='candidate_only',
                selected_epoch=best_epoch,training_horizon_mode='nominal6'),weights/f'{arm}_{seed}.pt')
            result['arms'][arm]=dict(tune_mae=best_mae,selected_epoch=best_epoch,log=log)
            if args.extended:
                if seed in (7,19,31):
                    pilot=json.loads(Path('artifacts/encoder_auxiliary_pilot_20260910.json').read_text())
                    previous=next(s for s in pilot['seeds'] if s['seed']==seed)['arms'][arm]['log']
                    if log[:8] != previous:
                        raise ValueError('first eight epochs differ from pilot')
                if arm=='glucose_only':
                    baseline=json.loads(Path('artifacts/encoder_source2250_64_20260910.json').read_text())
                    previous=next(s for s in baseline['seeds'] if s['seed']==seed)['training']['whole_unbounded']['log']
                    if any(a['tune_mae']!=b['tune_mae'] for a,b in zip(log,previous)) or len(log)!=len(previous):
                        raise ValueError('matched single-target baseline differs')
        for comparator in ('glucose_only','shuffled_auxiliary'):
            result['comparisons'][comparator]=paired(frame.loc[tune],
                predictions['auxiliary'],predictions[comparator],f'auxiliary-{seed}-{comparator}',1000)
        report['seeds'].append(result)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    if args.extended:
        report['integrity']=dict(pilot_first8_exact=True,single_target_all64_exact=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'SOURCE {stage.upper()} COMPLETE',flush=True)


if __name__=='__main__':
    main()

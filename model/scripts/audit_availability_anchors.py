"""Source-only trained diagnostic on newly availability-anchored glucose pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.audit_recurrent_event_state_loop import prepare_replay, patient_equal_mae
from scripts.audit_event_aware_patient_state import patient_partition, paired
from whole_body_jepa.multisource_intervals import SEEDS, sha256
from whole_body_jepa.availability_anchors import rebuild


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    args.model_dir = Path('models')
    args.scored_only = False
    root = Path('/Users/chunyouchang/Downloads')
    args.eicu = Path('/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.mimic = Path('/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.eicu_root = root/'eicu-collaborative-research-database-2.0'
    args.mimic_root = root/'mimic-iv-3.1'
    args.eicu_context_cache = Path('/private/tmp/event_aware_eicu_glucose_6h_context.npz')
    args.mimic_context_cache = Path('/private/tmp/event_aware_mimic_glucose_6h_context.npz')
    # Reuse ONLY the frozen subject-role assignment; discard old labels/predictions/features.
    roles, *_ = prepare_replay(args)
    data, support = {}, {}
    for role in ('development', 'eicu_hospital', 'mimic'):
        db = 'mimic' if role == 'mimic' else 'eicu'
        raw = pd.read_parquet(f'/private/tmp/raw_glucose_identity_{db}.parquet')
        ids = roles[role][['stay_id', 'subject_id']].drop_duplicates()
        if ids.stay_id.duplicated().any():
            raise ValueError('stay maps to multiple patients')
        f = rebuild(raw.loc[raw.stay_id.isin(ids.stay_id)]).merge(ids, on='stay_id', validate='many_to_one')
        f.to_parquet(f'/private/tmp/availability_glucose_{role}.parquet', index=False)
        valid = f.fresh & f.truth.notna()
        support[role] = {'queries': len(f), 'scored_rows': int(valid.sum()),
                         'scored_patients': int(f.loc[valid, 'subject_id'].nunique()),
                         'actual_delta_t_quantiles': f.loc[valid, 'actual_delta_t'].quantile([0,.5,1]).tolist(),
                         'current_available_after_anchor': int((f.current_available_at > f.t).sum()),
                         'label_before_anchor': int((f.future_event_time <= f.t).sum())}
        data[role] = f.loc[valid].reset_index(drop=True)
        print(json.dumps({role: support[role]}), flush=True)
    a, b, c = [set(data[r].subject_id) for r in data]
    if a & b or a & c or b & c:
        raise ValueError('overlapping patient roles')
    columns = ['current', 'measurement_age']
    history = [f'{name}_{h}' for h in (6,24) for name in ('mean','std','history_valid','slope')]
    dev = data['development']
    report = dict(status='candidate_only', promotion_allowed=False, runtime_modified=False,
                  causal_claim_allowed=False, untouched_lockbox=False,
                  scope='glucose-only availability reconstruction; not full JEPA validation',
                  old_features_reused=False, support=support, seeds=[],
                  binding={str(p): sha256(p) for p in (Path(__file__), Path('whole_body_jepa/availability_anchors.py'), Path('AVAILABILITY_ANCHOR_PROTOCOL_20260910.md'))})
    for seed in SEEDS:
        fit, tune = patient_partition(dev.subject_id, seed)
        if dev.loc[fit,'subject_id'].nunique()<50 or dev.loc[tune,'subject_id'].nunique()<30:
            raise ValueError('insufficient source support')
        models = {}
        for name, cols in [('current_only',columns), ('history',columns+history)]:
            scaler = StandardScaler().fit(dev.loc[fit,cols])
            x = scaler.transform(dev[cols])
            weights = dev.loc[fit].subject_id.map(dev.loc[fit].subject_id.value_counts()).rdiv(1.).to_numpy(copy=True)
            weights /= weights.mean()
            best = None
            for alpha in (1.,10.,100.):
                model = Ridge(alpha=alpha).fit(x[fit], (dev.truth-dev.current).to_numpy()[fit], sample_weight=weights)
                pred = dev.current.to_numpy()[tune]+model.predict(x[tune])
                score = patient_equal_mae(dev.loc[tune],dev.truth.to_numpy()[tune],pred)
                if best is None or score < best[0]:
                    best = (score,model,alpha)
            models[name] = (scaler,best[1],cols,best[2])
        evaluation = {}
        for role in ('eicu_hospital','mimic'):
            f = data[role]
            pred = {'persistence':f.current.to_numpy()}
            for name,(scaler,model,cols,_) in models.items():
                pred[name] = f.current.to_numpy()+model.predict(scaler.transform(f[cols]))
            evaluation[role] = dict(mae={n:patient_equal_mae(f,f.truth,v) for n,v in pred.items()},
                comparisons={n:paired(f,pred['history'],pred[n],f'availability:{role}:{seed}:{n}',1000) for n in ('persistence','current_only')})
        result = dict(seed=int(seed), alpha={n:m[3] for n,m in models.items()}, evaluation=evaluation)
        report['seeds'].append(result)
        print(json.dumps(result),flush=True)
    report['accuracy_comparison_complete'] = True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

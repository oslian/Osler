"""Fixed-source treatment-vs-lab-only diagnostic, not prospective deployment."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from extractors.eicu_body_system_configs import get_body_system_config
from extractors.eicu_body_system_transition_extract import classify_action
from extractors.eicu_aki_transition_extract import classify_aki_action
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.availability_inputs import build_arrays, restrict_input_support
from whole_body_jepa.event_window_history import event_windows, select_event_features
from whole_body_jepa.event_treatment import rate_channels
from whole_body_jepa.potassium_treatment_context import ACTIONS, context_features, shuffled_context, order_record, action_support_status, rate_history_features
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orders', action='store_true')
    parser.add_argument('--rate-history', action='store_true')
    parser.add_argument('--inspect-rate-history', action='store_true')
    args = parser.parse_args()
    if args.inspect_rate_history:args.rate_history=True
    if args.orders and args.rate_history:raise ValueError('orders have no administered rate history')
    name = 'potassium_order_context_20260911' if args.orders else 'potassium_treatment_context_20260911'
    protocol = 'POTASSIUM_ORDER_CONTEXT_PROTOCOL_20260911.md' if args.orders else 'POTASSIUM_TREATMENT_CONTEXT_PROTOCOL_20260911.md'
    if args.rate_history:
        name='potassium_rate_history_20260911'
        protocol='POTASSIUM_RATE_HISTORY_PROTOCOL_20260911.md'
    output = Path(f'artifacts/{name}.json')
    frozen_report_path=output
    if args.inspect_rate_history:output=Path('artifacts/potassium_rate_subgroup_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    root = Path('/private/tmp'); private = root/name
    private.mkdir(exist_ok=True)
    label = root/'auxiliary_supervision_20260910.npz'
    raw_paths = [root/'availability_jepa_eicu_labs.parquet', root/'encoder_source_expansion_20260910/labs.parquet']
    baseline_path = Path('artifacts/potassium_event_windows_20260911.json')
    baseline = json.loads(baseline_path.read_text())
    prior_path = Path('artifacts/potassium_tree_history_20260911.json')
    prior = json.loads(prior_path.read_text())
    for p in [label, prior_path]+raw_paths:
        if sha256(p) != baseline['binding'][str(p)]:
            raise ValueError('source binding changed')
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; col = variables.index('potassium')
    y = np.load(label); valid = y['paired'][:, col]
    frame = pd.DataFrame(dict(stay_id=y['stay_id'][valid], subject_id=y['subject_id'][valid], t=y['query_time'][valid], truth=y['future'][valid, col]))
    frame['offset'] = (pd.to_datetime(frame.t)-pd.Timestamp('2100-01-01')).dt.total_seconds()/60
    current = y['current'][valid, col]; scale = assets.scaler.scales[col]
    raw = pd.concat([pd.read_parquet(p) for p in raw_paths], ignore_index=True)
    raw = raw[raw.stay_id.isin(frame.stay_id)]
    arrays, rows = build_arrays(frame, raw, assets, {s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    windows = event_windows(frame, raw, variables, assets.scaler.medians, assets.scaler.scales)
    infusion = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')/('medication.csv.gz' if args.orders else 'infusionDrug.csv.gz')
    ids = set(frame.stay_id.str.replace('eicu:', '', regex=False).astype(int))
    configs = [get_body_system_config(n) for n in ('electrolyte_acid_base', 'endocrine_stress')]
    records = []
    for chunk in pd.read_csv(infusion, chunksize=500000, low_memory=False):
        for row in chunk[chunk.patientunitstayid.isin(ids)].itertuples(index=False):
            offset = pd.to_numeric(row.drugorderoffset if args.orders else row.infusionoffset, errors='coerce')
            if not np.isfinite(offset): continue
            actions = set(classify_aki_action(row.drugname))
            for config in configs: actions.update(classify_action(config, row.drugname))
            for action in set(ACTIONS)&actions:
                if args.orders:
                    records.append(order_record(row._asdict(), action))
                    continue
                channels = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate', rate_like=row.drugrate, original_label=row.drugname), action)
                value = channels['action_rate']; dimension = channels['action_rate_dimension']
                if np.isfinite(channels['solution_flow_ml_per_hour']):
                    value = channels['solution_flow_ml_per_hour']; dimension = 'volume_ml_per_hour'
                records.append(dict(stay_id=f'eicu:{row.patientunitstayid}', source_id=str(row.infusiondrugid), offset=float(offset), action=action, value=value, dimension=dimension, original_label=str(row.drugname)))
    events = pd.DataFrame(records).drop_duplicates(['source_id', 'action'])
    treatment, columns = context_features(frame, events)
    if args.rate_history:
        extra,extra_columns=rate_history_features(frame,events)
        treatment=np.concatenate([treatment,extra],axis=1);columns+=extra_columns
    bindings = [label, prior_path, baseline_path, infusion, Path(__file__), Path(protocol), Path('whole_body_jepa/potassium_treatment_context.py'), Path('whole_body_jepa/event_treatment.py'), Path('extractors/eicu_body_system_configs.py'), Path('extractors/eicu_body_system_transition_extract.py'), Path('extractors/eicu_aki_transition_extract.py')]+raw_paths
    report = dict(complete=False, status='retrospective_event_time_source_diagnostic', runtime_modified=False, promotion_allowed=False, external_data_read=False, availability_verified=False, columns=columns, binding={str(p):sha256(p) for p in bindings}, seeds=[])
    report['context_kind'] = 'order_issuance_not_administration' if args.orders else 'infusion_snapshot'
    if args.rate_history:
        report['context_kind']='infusion_snapshot_and_same_label_rate_history'
        report['rate_history_support']={str(c):int(np.isfinite(extra[:,i]).sum()) for i,c in enumerate(extra_columns)}
    frozen_report=json.loads(frozen_report_path.read_text()) if args.inspect_rate_history else None
    if frozen_report:
        report['binding'][str(frozen_report_path)]=sha256(frozen_report_path)
        report['status']='frozen_preanchor_subgroup_diagnostic'
        report['fitting_performed']=False
        report['rate_variation']={str(c):int((np.isfinite(extra[:,i]) & (np.abs(extra[:,i])>1e-8)).sum()) for i,c in enumerate(extra_columns)}
    delta = (frame.truth.to_numpy()-current)/scale
    for entry, reference in zip(prior['seeds'], baseline['seeds']):
        seed = entry['seed']; assert seed == reference['seed']
        masks = three_way_partition(frame.subject_id, np.ones(len(frame), bool), seed)
        fit, tune, evaluation = masks
        old_path = root/f'potassium_event_windows_20260911/seed_{seed}.joblib'
        if sha256(old_path) != baseline['binding'][str(old_path)]: raise ValueError('baseline changed')
        report['binding'][str(old_path)] = sha256(old_path)
        old = joblib.load(old_path)
        view = restrict_input_support(arrays, old['allowed'], assets.model.variable_module_ids.numpy())
        placebo = shuffled_context(frame, treatment, masks, seed)
        subjects = frame.loc[fit, 'subject_id']; weights = 1/subjects.groupby(subjects).transform('size').to_numpy(float); weights *= len(weights)/weights.sum()
        points = {'persistence':current[evaluation]}; models = {}
        for mode in ('own', 'full'):
            base = np.concatenate([history_features(view, rows, col, mode).astype(float), select_event_features(windows, old['allowed'], col, mode)], axis=1)
            points[mode] = current[evaluation]+scale*old['models'][mode].predict(base[evaluation])
            mae = patient_equal_mae(frame.loc[evaluation], frame.loc[evaluation, 'truth'], points[mode])
            if abs(mae-reference['evaluation']['mae'][mode]) > 1e-7: raise ValueError('baseline reproduction failed')
            for name, extra in (('real', treatment), ('shuffled', placebo)):
                x = np.concatenate([base, extra], axis=1)
                model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, learning_rate=.05, max_leaf_nodes=entry['selection'][mode]['max_leaf_nodes'], min_samples_leaf=30, l2_regularization=10., max_bins=63, early_stopping=False, random_state=seed)
                if not args.inspect_rate_history:model.fit(x[fit], delta[fit], sample_weight=weights)
                key = f'{mode}_{name}'; models[key] = model
                if args.inspect_rate_history:
                    saved=private/f'seed_{seed}.joblib'
                    if sha256(saved)!=frozen_report['binding'][str(saved)]:raise ValueError('frozen weights changed')
                    model=joblib.load(saved)['models'][key]
                    models[key]=model
                points[key] = current[evaluation]+scale*model.predict(x[evaluation])
        f = frame.loc[evaluation]
        comparisons = {f'{m}_real_vs_{b}':paired(f, points[f'{m}_real'], points[b], f'treatment-{seed}-{m}-{b}', 1000) for m in ('own','full') for b in (m, f'{m}_shuffled', 'persistence')}
        dest = private/f'seed_{seed}.joblib'
        if not args.inspect_rate_history:joblib.dump(dict(models=models, columns=columns, allowed=old['allowed']), dest)
        report['binding'][str(dest)] = sha256(dest)
        support={action:action_support_status(frame.subject_id, treatment[:,columns.index((action,'count'))]>0, masks) for action in ACTIONS}
        report['seeds'].append(dict(seed=seed, patients=int(f.subject_id.nunique()), rows=len(f), treatment_support=support, mae={k:patient_equal_mae(f, f.truth, p) for k,p in points.items()}, comparisons=comparisons))
        if args.inspect_rate_history:
            stored=next(s for s in frozen_report['seeds'] if s['seed']==seed)
            for key in points:
                if abs(report['seeds'][-1]['mae'][key]-stored['mae'][key])>1e-7:raise ValueError('frozen prediction reproduction failed')
            known=np.isfinite(extra[:,extra_columns.index(('insulin','drug_units_per_hour','change'))])[evaluation]
            report['seeds'][-1]['subgroups']={}
            for label,mask in [('known_insulin_sequence',known),('without_sequence',~known)]:
                sub=f.loc[mask];p={k:v[mask] for k,v in points.items()}
                report['seeds'][-1]['subgroups'][label]=dict(rows=len(sub),patients=int(sub.subject_id.nunique()),mae={k:patient_equal_mae(sub,sub.truth,v) for k,v in p.items()},comparisons={f'{m}_real_vs_{b}':paired(sub,p[f'{m}_real'],p[b],f'ratesub-{seed}-{label}-{m}-{b}',1000) for m in ('own','full') for b in (m,f'{m}_shuffled')})
        print(f'Treatment seed {seed} completed', flush=True)
    report['complete'] = True
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()

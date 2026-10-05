"""Component-held-out support decision probe for fixed nonlinear readouts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.nowcast import related_siblings


PROBES = ('constant', 'aggregate', 'variable', 'shuffled')
SEED = 20261003


def support_features(frame, variables, target):
    mask = frame[[f'observed_{v}' for v in variables]].to_numpy()
    age = frame[[f'age_{v}' for v in variables]].to_numpy(float)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid observation mask')
    mask = mask.astype(bool)
    hidden = [i for i,v in enumerate(variables) if v in ({target} | related_siblings(target))]
    if (mask[:,hidden].any() or not np.isfinite(age[mask]).all()
            or (age[mask] < -1e-6).any() or (age[mask] > 49+1e-5).any()
            or not mask.any(axis=1).all()):
        raise ValueError('Invalid masked measurement-age support')
    measured_age = np.where(mask, np.maximum(age, 0), 0.)
    counts = mask.sum(axis=1)
    aggregate = np.column_stack((counts, (mask & (age<=1+1e-6)).sum(axis=1),
        (mask & (age<=6+1e-6)).sum(axis=1), measured_age.sum(axis=1)/counts))
    variable = np.column_stack((mask.astype(float), np.log1p(measured_age)))
    return aggregate, variable


def ridge_predict(x, y, weights, query):
    if not all(np.isfinite(a).all() for a in (x, y, weights, query)):
        raise ValueError('Nonfinite decision inputs')
    scaler = StandardScaler().fit(x, sample_weight=weights)
    z = scaler.transform(x)
    q = scaler.transform(query)
    if not np.isfinite(z).all() or not np.isfinite(q).all():
        raise ValueError('Invalid fitted scaling')
    return Ridge(alpha=100., solver='cholesky').fit(z, y, sample_weight=weights).predict(q)


def crossfit(frame, aggregate, variable):
    if (frame.empty or frame.patient.duplicated().any() or frame.fold.nunique()!=1
            or frame.target.nunique()!=1 or not frame.role.eq('selection').all()
            or not np.isfinite(frame[['truth','native','body_probe','population']]).all().all()
            or aggregate.shape!=(len(frame),4) or variable.shape!=(len(frame),106)):
        raise ValueError('Invalid source-only query')
    gain = (frame.body_probe-frame.truth).abs().to_numpy()-(frame.native-frame.truth).abs().to_numpy()
    result = frame[['patient','component','fold','target','truth','native','body_probe','population']].copy()
    result['difference'] = gain
    for name in PROBES: result[f'expected_{name}'] = np.nan
    components = frame.component.to_numpy()
    people = frame.patient.to_numpy()
    diagnostics = []
    with threadpool_limits(limits=4):
        for group in sorted(set(components)):
            held = components==group
            train = ~held
            support = dict(component=str(group), query_patients=int(held.sum()),
                training_patients=int(train.sum()), training_components=len(set(components[train])))
            if train.sum()<30 or support['training_components']<2:
                diagnostics.append(dict(**support,status='insufficient_training_support'))
                continue
            weights = bounded_weights(components[train])
            result.loc[held,'expected_constant'] = np.average(gain[train], weights=weights)
            result.loc[held,'expected_aggregate'] = ridge_predict(aggregate[train],gain[train],weights,aggregate[held])
            result.loc[held,'expected_variable'] = ridge_predict(variable[train],gain[train],weights,variable[held])
            donor = donor_indices(people[train],components[train],SEED)
            result.loc[held,'expected_shuffled'] = ridge_predict(variable[train][donor],gain[train],weights,variable[held])
            diagnostics.append(dict(**support,status='complete',unchanged_shuffle_singletons=int(sum(donor==np.arange(len(donor))))))
    result['supported'] = result[[f'expected_{p}' for p in PROBES]].notna().all(axis=1)
    for name in PROBES:
        # Unavailable decisions stay explicitly unsupported; native is only a bookkeeping anchor.
        result[f'point_{name}'] = np.where(result[f'expected_{name}']<0, result.body_probe,result.native)
    return result, diagnostics


def assess(frame, seed, repetitions=1000):
    if (frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max()!=1):
        raise ValueError('Duplicated or ambiguous paired identity')
    names = ('native','body_probe','population', *PROBES)
    losses = pd.DataFrame({a:abs(frame[a if a in names[:3] else 'point_'+a]-frame.truth) for a in names})
    losses['patient'] = frame.patient.to_numpy()
    error = losses.groupby('patient')[list(names)].mean()
    error['component'] = frame.drop_duplicates('patient').set_index('patient').component.reindex(error.index)
    grouped = error.groupby('component')[list(names)].mean()
    comparisons = {}
    for i, ref in enumerate(('native','body_probe','constant','aggregate','shuffled')):
        paired = dict(patient=paired_summary(error.variable-error[ref],seed+i,repetitions),
            component=paired_summary(grouped.variable-grouped[ref],seed+100+i,repetitions))
        passed = bool(frame.supported.all() and len(error)>=30 and len(grouped)>=3 and all(
            p['mean']<0 and p['p_less'] is not None and p['p_less']<.05/(5*47) for p in paired.values()))
        comparisons[ref] = dict(paired=paired,diagnostic_family_screen=passed)
    decisions = {}
    supported = frame.loc[frame.supported]
    for name in PROBES:
        if supported.empty:
            decisions[name] = None
            continue
        selected = (supported[f'expected_{name}']<0).groupby(supported.patient).mean()
        square = ((supported[f'expected_{name}']-supported.difference)**2).groupby(supported.patient).mean()
        group = supported.drop_duplicates('patient').set_index('patient').component
        decisions[name] = dict(chosen_body_patient_fraction=float(selected.mean()),
            expected_difference_mse_patient=float(square.mean()),
            expected_difference_mse_component=float(square.groupby(group).mean().mean()))
    return dict(patients=len(error),occurrences=len(frame),components=len(grouped),
        unsupported_occurrences=int((~frame.supported).sum()),
        normalized_mae=dict(patient=error[list(names)].mean().to_dict(),component=grouped.mean().to_dict()),
        comparisons=comparisons,decisions=decisions,
        per_component=[dict(component=str(c),patients=int((error.component==c).sum()),normalized_mae=r.to_dict()) for c,r in grouped.iterrows()])


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    dest = root/'nonlinear_probe_20261001/support_selector_20261003'
    out = Path('artifacts/native_reconstruction_nonlinear_support_selector_20261003.json')
    if out.exists(): raise ValueError('Completed probe exists')
    dest.mkdir(exist_ok=True)
    parent_path = Path('artifacts/native_reconstruction_nonlinear_source_components_20261003.json')
    parent = json.loads(parent_path.read_text())
    assessment_path = Path('artifacts/native_reconstruction_nonlinear_placebo_assessment_20261003.json')
    assessment = json.loads(assessment_path.read_text())
    checked = {}
    verify_bound_files(parent['bindings'],checked)
    snapshot_path = root/'variable_snapshot_20260925.parquet'
    snapshot_report = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    metadata = json.loads(snapshot_report.read_text())
    verify_bound_files(metadata['bindings'],checked)
    if sha256(snapshot_path)!=metadata['private_sha256']: raise ValueError('Changed masked input snapshot')
    variables = metadata['variables']
    if len(variables)!=53: raise ValueError('Changed input vocabulary')
    plan_path, provenance = root/'partitions_20260922.pt',root/'provenance.parquet'
    plan = torch.load(plan_path,weights_only=False)['folds']
    links = pd.read_parquet(provenance)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous components')
    mapping = dict(links.itertuples(index=False,name=None))
    for roles in plan:
        component_roles = {}
        for p,role in roles.items(): component_roles.setdefault(mapping[p],set()).add(role)
        if any(len(r)!=1 for r in component_roles.values()): raise ValueError('Component role overlap')
    paths = [Path(__file__),Path('RECONSTRUCTION_SUPPORT_SELECTOR_PROTOCOL_20261003.md'),parent_path,
        snapshot_report,snapshot_path,plan_path,provenance,assessment_path,Path('scripts/bounded_site_sampling.py'),
        Path('scripts/train_nonlinear_placebo.py'),Path('scripts/score_additional_reconstruction.py'),
        Path('whole_body_jepa/nowcast.py')]
    bindings = {str(p):sha256(p) for p in paths}
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=bindings: raise ValueError('Changed resume contract')
    if not cp.exists(): cp.write_text(json.dumps(bindings,indent=2)+'\n')
    all_results = {}
    for i,target in enumerate(sorted(parent['targets'])):
        sp = dest/f'{target}.json'
        qp = dest/f'{target}.parquet'
        if sp.exists():
            saved = json.loads(sp.read_text())
            if saved['contract_sha256']!=sha256(cp) or saved['prediction_sha256']!=sha256(qp):
                raise ValueError('Changed completed target')
            all_results[target] = saved
            continue
        columns = ['patient','target',*[f'observed_{v}' for v in variables],*[f'age_{v}' for v in variables]]
        snapshot = pd.read_parquet(snapshot_path,columns=columns,filters=[('target','==',target)])
        if snapshot.patient.duplicated().any(): raise ValueError('Duplicate input identity')
        frames, folds = [], {}
        for fold in range(5):
            p = root/f'nonlinear_probe_20261001/placebo_20261003/fold{fold}/{target}.parquet'
            if sha256(p)!=parent['bindings'].get(str(p)): raise ValueError('Changed paired outcomes')
            frame = pd.read_parquet(p,filters=[('role','==','selection')])
            if any(plan[fold][p]!='selection' for p in frame.patient): raise ValueError('Wrong source roles')
            if not np.array_equal(frame.component,frame.patient.map(mapping)): raise ValueError('Changed provenance')
            merged = frame.merge(snapshot,on=['patient','target'],how='left',validate='one_to_one',indicator=True)
            if not merged._merge.eq('both').all(): raise ValueError('Missing input support')
            aggregate, variable = support_features(merged,variables,target)
            predicted, diagnostics = crossfit(merged,aggregate,variable)
            frames.append(predicted); folds[str(fold)] = diagnostics
        combined = pd.concat(frames,ignore_index=True)
        result = assess(combined,SEED+1000*i)
        expected = assessment['results']['selection'][target]
        if result['patients']!=expected['unique_patients'] or result['components']!=expected['components']:
            raise ValueError('Changed source support')
        for level in ('patient','component'):
            for arm in ('native','body_probe','population'):
                if not np.isclose(result['normalized_mae'][level][arm],expected['normalized_mae'][level][arm],rtol=1e-10,atol=1e-10):
                    raise ValueError('Original error parity failed')
        combined.to_parquet(qp,index=False)
        saved = dict(result=result,folds=folds,contract_sha256=sha256(cp),prediction_sha256=sha256(qp))
        sp.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
        all_results[target] = saved
        print('Completed source support selector',i+1,'of',len(parent['targets']),flush=True)
    if len(all_results)!=47: raise ValueError('Incomplete target family')
    summary = dict(full_five_contrast_diagnostic_pass=[t for t,r in all_results.items() if all(c['diagnostic_family_screen'] for c in r['result']['comparisons'].values())],
        macro_normalized_mae={level:{arm:float(np.mean([r['result']['normalized_mae'][level][arm] for r in all_results.values()])) for arm in ('native','body_probe','population',*PROBES)} for level in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,targets=all_results,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='All47/five source folds. Component-held-out decision probe on fixed source out-of-fit predictions, not a new external or deployment gate. Mask/age-only features are medical process signals, not physiological state proof. Patient losses average overlapping folds; no independent replication claim. Single fixed shuffled null. Unsupported decisions remain marked and cannot pass. No outer selection, encoder/readout changes or intervals.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

"""Same-state readout complementarity with component-held-out convex fitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from scripts.diagnose_nonlinear_fit_gap import weighted_median
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


HEADS = ('body_probe','process_summary','trained_shuffle')
ARMS = ('native','population',*HEADS,'equal_blend',*(a+'_blend' for a in HEADS))
REFERENCES = ('native','body_probe','population','equal_blend','process_summary_blend','trained_shuffle_blend')


def fit_weight(truth, native, proposal, components):
    y,a,b = [np.asarray(v,float) for v in (truth,native,proposal)]
    groups = np.asarray(components)
    if (y.ndim!=1 or not len(y) or a.shape!=y.shape or b.shape!=y.shape or groups.shape!=y.shape
            or not all(np.isfinite(v).all() for v in (y,a,b)) or pd.isna(groups).any()):
        raise ValueError('Invalid weight-fitting support')
    difference = b-a
    changed = difference!=0
    if not changed.any(): return 0.
    counts = pd.Series(groups).value_counts()
    mass = np.array([1/counts[g] for g in groups])
    # Outside [0,1], clipping ratios preserves the constrained weighted median.
    # Clip the numerator first to avoid overflow for extremely tiny proposals.
    residual = np.clip(y-a,np.minimum(0,difference),np.maximum(0,difference))
    ratios = residual[changed]/difference[changed]
    weights = mass[changed]*abs(difference[changed])
    return float(np.clip(weighted_median(ratios,weights),0,1))


def validate_frame(frame):
    if (frame.empty or frame.patient.duplicated().any() or frame.target.nunique()!=1
            or frame.fold.nunique()!=1 or not frame.role.eq('selection').all()
            or frame.component.isna().any()
            or not np.isfinite(frame[['truth','native','population',*HEADS]]).all().all()):
        raise ValueError('Invalid source-only paired frame')


def crossfit(frame):
    validate_frame(frame)
    result = frame[['patient','component','target','fold','role','truth','native','population',*HEADS]].copy()
    result['equal_blend'] = (result.native+result.body_probe)/2
    result['supported'] = False
    for arm in HEADS:
        result[arm+'_weight'] = np.nan
        result[arm+'_blend'] = result.native
    groups = result.component.to_numpy()
    diagnostics = []
    for group in sorted(set(groups)):
        held = groups==group
        train = ~held
        record = dict(component=str(group),query_patients=int(held.sum()),
                      training_patients=int(train.sum()),training_components=len(set(groups[train])))
        if train.sum()<30 or record['training_components']<2:
            diagnostics.append(dict(**record,status='insufficient_training_support',weights=None))
            continue
        fitting = result.loc[train]
        weights = {}
        for arm in HEADS:
            w = fit_weight(fitting.truth,fitting.native,fitting[arm],fitting.component)
            result.loc[held,arm+'_weight'] = w
            result.loc[held,arm+'_blend'] = (1-w)*result.loc[held,'native']+w*result.loc[held,arm]
            weights[arm] = w
        result.loc[held,'supported'] = True
        diagnostics.append(dict(**record,status='complete',weights=weights))
    return result,diagnostics


def correlation(a,b,rank=False):
    a,b = np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0: return None
    return float(spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1])


def assess(frame,seed,repetitions=2000):
    if (frame.empty or frame.target.nunique()!=1 or not frame.role.eq('selection').all()
            or frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth',*ARMS]]).all().all()):
        raise ValueError('Invalid paired blend assessment')
    loss = frame[list(ARMS)].subtract(frame.truth,axis=0).abs()
    loss['patient'] = frame.patient.to_numpy()
    patient = loss.groupby('patient')[list(ARMS)].mean()
    grouping = frame.drop_duplicates('patient').set_index('patient').component
    patient['component'] = grouping.reindex(patient.index)
    component = patient.groupby('component')[list(ARMS)].mean()
    comparisons = {}
    for i,ref in enumerate(REFERENCES):
        paired = dict(patient=paired_summary(patient.body_probe_blend-patient[ref],seed+i,repetitions),
            component=paired_summary(component.body_probe_blend-component[ref],seed+100+i,repetitions))
        passed = bool(frame.supported.all() and len(patient)>=30 and len(component)>=3 and all(
            p['mean']<0 and p['p_less'] is not None and p['p_less']<.05/(47*len(REFERENCES))
            and p['ci95'][1]<0 for p in paired.values()))
        comparisons[ref] = dict(paired=paired,diagnostic_family_screen=passed)
    e1,e2 = frame.native-frame.truth,frame.body_probe-frame.truth
    cancellations = (abs(e1)+abs(e2))/2-abs((e1+e2)/2)
    if (cancellations < -1e-10).any(): raise ValueError('Convex cancellation invariant failed')
    paired_errors = pd.DataFrame(dict(patient=frame.patient,native=e1,body=e2,
        opposite=((e1<0)&(e2>0))|((e1>0)&(e2<0)),cancellation=cancellations))
    per_patient = paired_errors.groupby('patient')[['native','body','opposite','cancellation']].mean()
    per_component = per_patient.groupby(grouping).mean()
    complementarity = dict(patient_signed_error_pearson=correlation(per_patient.native,per_patient.body),
        patient_signed_error_spearman=correlation(per_patient.native,per_patient.body,True),
        opposite_error_fraction_patient=float(per_patient.opposite.mean()),
        opposite_error_fraction_component=float(per_component.opposite.mean()),
        cancellation_vs_average_member_loss_patient=float(per_patient.cancellation.mean()),
        cancellation_vs_average_member_loss_component=float(per_component.cancellation.mean()))
    weights = {}
    part = frame.loc[frame.supported]
    for arm in HEADS:
        w = part[arm+'_weight']
        weights[arm] = None if part.empty else dict(mean_patient=float(w.groupby(part.patient).mean().mean()),
            interior_fraction_patient=float(((w>0)&(w<1)).groupby(part.patient).mean().mean()),
            zero_fraction_patient=float((w==0).groupby(part.patient).mean().mean()),
            one_fraction_patient=float((w==1).groupby(part.patient).mean().mean()))
    return dict(patients=len(patient),occurrences=len(frame),components=len(component),
        unsupported_occurrences=int((~frame.supported).sum()),
        normalized_mae=dict(patient=patient[list(ARMS)].mean().to_dict(),component=component.mean().to_dict()),
        comparisons=comparisons,complementarity=complementarity,weights=weights,
        per_component=[dict(component=str(c),patients=int((patient.component==c).sum()),normalized_mae=r.to_dict())
                       for c,r in component.iterrows()])


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    features = root/'nonlinear_probe_20261001'
    dest = features/'blend_20261003'
    out = Path('artifacts/native_reconstruction_nonlinear_blend_20261003.json')
    if out.exists(): raise ValueError('Completed blend audit exists')
    dest.mkdir(exist_ok=True)
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    gap_path = Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    source,gap = [json.loads(p.read_text()) for p in (source_path,gap_path)]
    checked = {}; verify_bound_files(source['bindings'],checked)
    targets = sorted(source['targets'])
    if len(targets)!=47: raise ValueError('Changed target family')
    pp, prov_path = root/'partitions_20260922.pt',root/'provenance.parquet'
    plan = torch.load(pp,weights_only=False)['folds']
    links = pd.read_parquet(prov_path)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    mapping = dict(links.itertuples(index=False,name=None))
    for roles in plan:
        groups = {}
        for p,r in roles.items(): groups.setdefault(mapping[p],set()).add(r)
        if any(len(r)!=1 for r in groups.values()): raise ValueError('Mixed component roles')
    paths = [Path(__file__),Path('RECONSTRUCTION_READOUT_BLEND_PROTOCOL_20261003.md'),source_path,gap_path,pp,prov_path,
             Path('scripts/diagnose_nonlinear_fit_gap.py'),Path('scripts/score_additional_reconstruction.py')]
    proofs = {}
    for fold in range(5):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
        if sha256(rp)!=gap['bindings'][str(rp)]: raise ValueError('Changed null proof')
        proofs[fold] = json.loads(rp.read_text()); paths.append(rp)
        if proofs[fold]['fold']!=fold or sorted(proofs[fold]['targets'])!=targets: raise ValueError('Wrong null family')
    bindings = {str(p):sha256(p) for p in paths}
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=bindings: raise ValueError('Changed blend resume contract')
    if not cp.exists(): cp.write_text(json.dumps(bindings,indent=2)+'\n')
    results = {}
    for i,target in enumerate(targets):
        sp, qp = dest/f'{target}.json',dest/f'{target}.parquet'
        if sp.exists():
            saved = json.loads(sp.read_text())
            if saved['contract_sha256']!=sha256(cp) or saved['prediction_sha256']!=sha256(qp): raise ValueError('Changed saved result')
            results[target] = saved; continue
        frames, decisions = [], {}
        for fold in range(5):
            path = features/f'placebo_20261003/fold{fold}/{target}.parquet'
            if sha256(path)!=proofs[fold]['targets'][target]['prediction_sha256']: raise ValueError('Changed prediction file')
            frame = pd.read_parquet(path,filters=[('role','==','selection')])
            validate_frame(frame)
            if (not frame.fold.eq(fold).all() or not frame.target.eq(target).all()
                    or any(plan[fold][p]!='selection' for p in frame.patient)
                    or not np.array_equal(frame.component,frame.patient.map(mapping))):
                raise ValueError('Changed source identities or provenance')
            prior_path = features/f'fold{fold}_probes/{target}.parquet'
            prior = pd.read_parquet(prior_path).set_index('patient').sort_index()
            paired = frame.set_index('patient').sort_index()
            if (not prior.index.equals(paired.index)
                    or not np.allclose(prior[['truth','native','body_probe','population','process_summary']],
                                       paired[['truth','native','body_probe','population','process_summary']],atol=1e-10,rtol=1e-10)):
                raise ValueError('Changed original matched predictions')
            predicted,records = crossfit(frame)
            frames.append(predicted); decisions[str(fold)] = records
        combined = pd.concat(frames,ignore_index=True)
        result = assess(combined,20261003+1000*i)
        original = source['targets'][target]
        if result['patients']!=original['unique_patients'] or result['occurrences']!=original['query_occurrences']:
            raise ValueError('Incomplete target support')
        for level in ('patient','component'):
            for arm in ('native','body_probe','population','process_summary'):
                if not np.isclose(result['normalized_mae'][level][arm],original['normalized_mae'][level][arm],atol=1e-10,rtol=1e-10):
                    raise ValueError('Original source means changed')
        tmp = qp.with_suffix('.tmp'); combined.to_parquet(tmp,index=False); tmp.replace(qp)
        saved = dict(result=result,decisions=decisions,contract_sha256=sha256(cp),prediction_sha256=sha256(qp))
        sp.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n'); results[target] = saved
        print('Completed all-fold blend target',i+1,'of47',flush=True)
    summary = dict(full_six_contrast_screen=[t for t,r in results.items() if all(c['diagnostic_family_screen'] for c in r['result']['comparisons'].values())],
        macro_normalized_mae={l:{a:float(np.mean([r['result']['normalized_mae'][l][a] for r in results.values()])) for a in ARMS} for l in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,targets=results,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='Same frozen shared-state readouts; all47/five source folds, component-held-out single blend weight. Reused development and overlapping fitted models; conditional intervals exclude refit/selection variability. Cancellation is retrospective and relative to average member loss, not a legal row oracle or independent signal proof. No new outer/forecast/causal/runtime/interval claim.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

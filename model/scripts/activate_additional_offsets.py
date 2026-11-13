"""Fixed calibration-only correction activation, evaluated without retuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from scripts.calibrate_additional_reconstruction import fit_offsets, compare
from scripts.diagnose_additional_offset_transport import leave_component_out, transfer_delta
from scripts.score_additional_reconstruction import ARMS
from whole_body_jepa.multisource_intervals import sha256


def decide(cv):
    values=np.array([r['delta'] for r in cv['groups'] if r['status']=='fitted'],dtype=float)
    if not cv['complete'] or len(values)<4:
        return dict(active=False,p_less=None,reason='insufficient_cv_support')
    mean=float(values.mean())
    p=(0. if mean<0 else 1.) if np.ptp(values)==0 else float(ttest_1samp(values,0,alternative='less').pvalue)
    active=bool(mean<0 and np.isfinite(p) and p<.05)
    return dict(active=active,p_less=p,reason='source_cv_gain' if active else 'source_cv_gain_unproven')


def source_decisions(frame):
    full=fit_offsets(frame);result={}
    for arm in ARMS:
        work=frame.copy();work['real']=frame[arm]
        cv=leave_component_out(work);decision=decide(cv)
        decision.update(offset=full['offsets'][arm] if decision['active'] else 0.,
                        full_offset=full['offsets'][arm],cv=cv)
        result[arm]=decision
    return result


def main():
    parent=Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    report=json.loads(parent.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed calibration dependency')
    out=Path('artifacts/native_reconstruction_offset_activation_20260925.json')
    if out.exists():raise ValueError('Existing activation result')
    root=Path('data/research_recovery/additional_development_domains_20260921')
    targets=sorted(report['targets']);decisions={};frames=[];activation_counts=dict.fromkeys(ARMS,0)
    enabled_outcomes=[]
    for fold in range(5):
        cal=pd.read_parquet(root/f'calibration_20260924/fold{fold}.parquet')
        outer=pd.read_parquet(root/f'outer_20260924/fold{fold}.parquet')
        if set(cal.patient)&set(outer.patient) or set(cal.component)&set(outer.component):raise ValueError('Role overlap')
        # Complete source-only decisions before applying to outer predictions.
        choices={t:source_decisions(cal.loc[cal.target==t]) for t in targets}
        decisions[str(fold)]=choices
        work=outer.copy();work['uncorrected_real']=work.real;work['unconditional_real']=work.real
        for t,choice in choices.items():
            mask=work.target==t
            for a in ARMS:
                activation_counts[a]+=int(choice[a]['active'])
                work.loc[mask,a]+=choice[a]['offset']
            work.loc[mask,'unconditional_real']+=report['offsets'][str(fold)][t]['offsets']['real']
            if choice['real']['active']:
                enabled_outcomes.append(dict(fold=fold,target=t,delta=transfer_delta(outer.loc[mask],choice['real']['offset'])))
        frames.append(work)
    data=pd.concat(frames,ignore_index=True)
    if data.duplicated(['patient','target']).any() or len(data)!=report['summary']['pairs']:raise ValueError('Outer identity changed')
    results={}
    for i,t in enumerate(targets):
        part=data.loc[data.target==t]
        result=compare(part,20260925+1000*i)
        delta=(part.real-part.truth).abs()-(part.unconditional_real-part.truth).abs()
        result['vs_unconditional_mean']=dict(patient=float(delta.mean()),component=float(delta.groupby(part.component).mean().mean()))
        results[t]=result
        print('Scored activation',i+1,'of',len(targets),flush=True)
    summary=dict(pairs=len(data),targets=len(targets),activation_counts=activation_counts,
        incremental_descriptive_passes=[t for t,r in results.items() if r['incremental_descriptive_screen']],
        normalized_target_macro={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
            for a in ('real','process','population','uncorrected_real')} for level in ('patient','component')},
        enabled_real_helpful_components=sum(r['delta'] is not None and r['delta']['component']<0 for r in enabled_outcomes),
        enabled_real_nonhelpful_components=sum(r['delta'] is not None and r['delta']['component']>=0 for r in enabled_outcomes))
    paths=[parent,Path(__file__),Path('scripts/calibrate_additional_reconstruction.py'),
        Path('scripts/diagnose_additional_offset_transport.py'),Path('scripts/score_additional_reconstruction.py'),
        Path('RECONSTRUCTION_OFFSET_ACTIVATION_PROTOCOL_20260925.md')]
    out.write_text(json.dumps(dict(summary=summary,decisions=decisions,targets=results,enabled_real_outcomes=enabled_outcomes,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Fixed source-CV activation heuristic; overlapping CV, reused outer validation, no independent confirmation or promotion. Current masked reconstruction only.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

"""Leave-component-out calibration diagnosis; never deploys an outer selector."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.calibrate_additional_reconstruction import fit_offsets
from whole_body_jepa.multisource_intervals import sha256


def transfer_delta(frame, offset):
    if frame.empty:return None
    delta=(frame.real+offset-frame.truth).abs()-(frame.real-frame.truth).abs()
    return dict(patient=float(delta.mean()),component=float(delta.groupby(frame.component).mean().mean()))


def leave_component_out(frame):
    rows=[]
    for component, test in frame.groupby('component',sort=True):
        fitted=fit_offsets(frame.loc[frame.component!=component])
        if fitted['status']!='fitted':
            rows.append(dict(component=str(component),patients=len(test),status='insufficient_support',offset=None,delta=None))
            continue
        offset=fitted['offsets']['real']
        rows.append(dict(component=str(component),patients=len(test),status='fitted',offset=offset,
                         delta=transfer_delta(test,offset)['patient']))
    valid=[r for r in rows if r['status']=='fitted']
    return dict(groups=rows,supported_groups=len(valid),total_groups=len(rows),
        complete=bool(rows) and len(valid)==len(rows),
        mean_delta=float(np.mean([r['delta'] for r in valid])) if valid else None,
        offset_range=[min(r['offset'] for r in valid),max(r['offset'] for r in valid)] if valid else None)


def main():
    path=Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    report=json.loads(path.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed calibration dependency')
    root=Path('data/research_recovery/additional_development_domains_20260921')
    results={};counts=dict(complete=0,source_cv_help_outer_help=0,source_cv_help_outer_harm=0,
        source_cv_nonhelp_outer_help=0,source_cv_nonhelp_outer_nonhelp=0,incomplete=0)
    for fold in range(5):
        cal=pd.read_parquet(root/f'calibration_20260924/fold{fold}.parquet')
        outer=pd.read_parquet(root/f'outer_20260924/fold{fold}.parquet')
        targets={}
        for t,saved in report['offsets'][str(fold)].items():
            source=cal.loc[cal.target==t];test=outer.loc[outer.target==t]
            if fit_offsets(source)!=saved:raise ValueError('Offset changed')
            cv=leave_component_out(source)
            outcome=transfer_delta(test,saved['offsets']['real'])
            targets[t]=dict(calibration_status=saved['status'],source_cv=cv,outer_delta=outcome)
            if saved['status']=='fitted' and cv['complete'] and outcome is not None:
                counts['complete']+=1
                source_help=cv['mean_delta']<0;outer_help=outcome['component']<0
                key=('source_cv_help_outer_help' if outer_help else 'source_cv_help_outer_harm') if source_help else (
                    'source_cv_nonhelp_outer_help' if outer_help else 'source_cv_nonhelp_outer_nonhelp')
                counts[key]+=1
            else:counts['incomplete']+=1
        results[str(fold)]=targets
        print('Completed calibration transport fold',fold,flush=True)
    out=Path('artifacts/native_reconstruction_additional_offset_transport_20260925.json')
    if out.exists():raise ValueError('Existing diagnosis')
    out.write_text(json.dumps(dict(summary=counts,folds=results,
        bindings={str(path):sha256(path),str(Path(__file__)):sha256(Path(__file__))},
        runtime_modified=False,promotion_allowed=False,
        limitation='Retrospective diagnosis, not selection or gate. Source leave-component-out correction excludes that component. Outer signs used only to describe transport. No corrected candidate is enabled from these signs.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()

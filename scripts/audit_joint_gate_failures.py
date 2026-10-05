"""Explain fixed gate failures without changing selection or acceptance rules."""
from collections import Counter
import json
from pathlib import Path
from whole_body_jepa.multisource_intervals import sha256


def reasons(comparison):
    result=[]
    if comparison['mean']>=0:result.append('pooled_mean_not_better')
    if comparison['site_mean']>=0:result.append('equal_site_mean_not_better')
    if comparison['sites']<3:result.append('insufficient_sites')
    if comparison['patient_p']>=.05/39:result.append('patient_evidence_insufficient')
    if comparison['site_p']>=.05/39:result.append('site_evidence_insufficient')
    return result


def main():
    src=Path('artifacts/native_reconstruction_joint_source_evaluation_20260918.json')
    out=Path('artifacts/native_reconstruction_joint_gate_failures_20260918.json')
    if out.exists():raise ValueError('Existing audit')
    report=json.loads(src.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Evidence changed')
    result={};summary={}
    for group,targets in report['groups'].items():
        per={};counts=Counter();refs={}
        for target,row in targets.items():
            if 'mae' not in row:
                per[target]=dict(status='no_trained_source_head');counts['no_trained_source_head']+=1;continue
            checks={k:reasons(v) for k,v in row['comparisons'].items()}
            core={k:v for k,v in checks.items()if k!='broad_only'}
            flat=set(x for values in core.values()for x in values)
            if not row['source_supported']:flat.add('source_support_insufficient')
            if row['patients']<30:flat.add('patient_support_insufficient')
            counts.update(flat)
            for k,v in checks.items():refs.setdefault(k,Counter()).update(v)
            per[target]=dict(source_supported=row['source_supported'],comparisons=checks,
                core_failure_reasons=sorted(flat),
                positive_means_all_core=all(row['comparisons'][k]['mean']<0 and row['comparisons'][k]['site_mean']<0 for k in core),
                core_pass=not flat,reported_pass=row['development_gate'])
        result[group]=per
        summary[group]=dict(failure_counts=dict(counts),reference_failure_counts={k:dict(v)for k,v in refs.items()},
            source_supported_positive_all_core=[t for t,r in per.items()if r.get('source_supported')and r.get('positive_means_all_core')],
            core_pass=[t for t,r in per.items()if r.get('core_pass')])
    out.write_text(json.dumps(dict(groups=result,summary=summary,
        bindings={str(p):sha256(p)for p in (Path(__file__),src)},
        limitation='Descriptive gate decomposition; insufficient evidence is not proof of no signal, nor permission to loosen gates.',
        runtime_modified=False,promotion_allowed=False),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

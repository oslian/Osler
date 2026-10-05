"""Frozen bias-polish failure localization using the existing site audit."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_continued_site_transfer import site_summary
from scripts.audit_continued_selection_pairs import interval
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    rp=Path('artifacts/native_reconstruction_bias_polish_20260920.json')
    er=Path('artifacts/native_reconstruction_bounded_site_evaluation_20260920.json')
    cp=root/'bias_polish_20260920/candidate.pt'
    out=Path('artifacts/native_reconstruction_bias_polish_sites_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    report,original=[json.loads(p.read_text()) for p in (rp,er)]
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(cp)!=report['candidate_sha256']:raise ValueError('Candidate changed')
    state=torch.load(cp,weights_only=False)['state_dict']
    parent=torch.load(root/'bounded_site_training/bounded.pt',weights_only=False)
    variables=parent['variables'];results={};summaries={}
    for group,per in report['groups'].items():
        p=root/f'bounded_site_evaluation/{group}.pt'
        if sha256(p)!=original['bindings'][str(p)]:raise ValueError('Evaluation changed')
        rows=torch.load(p,weights_only=False);result={}
        for target,previous in per.items():
            chosen=[r for r in rows if r['target']==target]
            col=variables.index(target);shift=float(state['readout.bias'][col]-parent['best_state_dict']['readout.bias'][col])
            delta=np.array([abs(r['site_bounded']+shift-r['truth'])-abs(r['site_bounded']-r['truth']) for r in chosen])
            sites=np.array([r['site'] for r in chosen]);summary=site_summary(delta,sites)
            if not np.isclose(summary['pooled'],previous['comparison']['mean'],atol=1e-10,rtol=0):raise ValueError('Mismatch')
            details={str(site):interval(delta[sites==site]) if (sites==site).sum()>=30 else dict(patients=int((sites==site).sum()),mean=float(delta[sites==site].mean()),lower=None,upper=None) for site in sorted(set(sites))}
            reasons=[];c=previous['comparison']
            for label,fail in [('patient_mean',c['mean']>=0),('site_mean',c['site_mean']>=0),('patient_evidence',c['patient_p']>=.05/39),('site_evidence',c['site_p']>=.05/39)]:
                if fail:reasons.append(label)
            result[target]=dict(**summary,failures=reasons,site_details=details,
                supported_nominal_harm_sites=[s for s,v in details.items() if v['lower'] is not None and v['lower']>0])
        results[group]=result
        summaries[group]=dict(failures=dict(Counter(reason for r in result.values() for reason in r['failures'])),
            mean_better_but_evidence_failed=[t for t,r in result.items() if r['pooled']<0 and r['equal_site']<0 and r['failures']],
            targets_with_supported_nominal_harm=[t for t,r in result.items() if r['supported_nominal_harm_sites']],
            weighting_sign_flips=[t for t,r in result.items() if r['weighting_sign_disagreement']])
    out.write_text(json.dumps(dict(groups=results,summary=summaries,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),rp,er,cp)},
        limitation='Reused fixed evaluation, original gate unchanged. Site intervals unadjusted descriptive and only n>=30; nominal harm is not multiplicity-adjusted harm. No site removal or model selection.'),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

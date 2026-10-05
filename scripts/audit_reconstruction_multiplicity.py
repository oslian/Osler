"""Post-hoc multiplicity sensitivity on frozen patient-level paired errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import permutation_test
from whole_body_jepa.multisource_intervals import sha256


def main():
    source=Path('artifacts/native_reconstruction_process_evaluation_20260914.json')
    parent=json.loads(source.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed dependency')
    private=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_process_training_20260914/heldout_predictions.pt')
    rows=torch.load(private,weights_only=False)
    targets=sorted(set.union(*(set(v) for v in parent['roles'].values())))
    family=len(targets)*len(rows)*2
    results={}
    for role,records in rows.items():
        frame=pd.DataFrame(records);results[role]={}
        for target in targets:
            part=frame.loc[frame.target.eq(target)]
            if part.subject_id.duplicated().any():raise ValueError('requires one pair per patient-target')
            comparisons={}
            for ref in ('population','process'):
                d=(abs(part.real-part.truth)-abs(part[ref]-part.truth)).to_numpy()
                if len(d)<30:
                    comparisons[ref]=dict(patients=len(d),supported=False,p_adjusted=1.,pass_sensitivity=False);continue
                expected=parent['roles'][role][target]['comparisons'][ref]['real']['mean']
                if not np.isclose(d.mean(),expected,rtol=0,atol=1e-10):raise ValueError('paired errors mismatch')
                test=permutation_test((d,),lambda x,axis:np.mean(x,axis=axis),
                    permutation_type='samples',vectorized=True,n_resamples=99999,batch=256,
                    alternative='less',rng=np.random.default_rng(20260914))
                p=float(test.pvalue);adjusted=min(1.,p*family)
                comparisons[ref]=dict(patients=len(d),supported=True,mean_delta=float(d.mean()),
                    p_unadjusted=p,p_adjusted=adjusted,pass_sensitivity=bool(adjusted<.05))
            results[role][target]=comparisons
        print(role,'complete',flush=True)
    retained=[t for t in targets if all(results[r][t][ref]['pass_sensitivity'] for r in rows for ref in ('population','process'))]
    output=Path('artifacts/native_reconstruction_multiplicity_20260914.json')
    if output.exists():raise ValueError('existing evidence')
    output.write_text(json.dumps(dict(complete=True,family_size=family,method='one-sided paired sign-flip Monte Carlo p, Bonferroni across complete family',
        assumption='Exchangeable model-error signs under the null; symmetry can fail. Post-hoc diagnostic, not independent confirmation or distribution-free mean-error inference.',
        roles=results,retained_targets=retained,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),source,private)}),indent=2)+'\n')
    print('retained',retained,flush=True)


if __name__=='__main__':main()

"""Check calibration-selection optimism using identically weighted metrics."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from whole_body_jepa.multisource_intervals import sha256


def comparison(choice,groups):
    anticipated=choice['cv_scores'][choice['name']]-choice['cv_scores']['unchanged']
    realized=float(np.mean([v['comparison']['site_mean'] for v in groups.values()]))
    return dict(changed=choice['name']!='unchanged',cv_delta=anticipated,held_site_delta=realized,
        optimism=realized-anticipated,held_group_deltas={g:v['comparison']['site_mean'] for g,v in groups.items()})


def main():
    rp=Path('artifacts/native_reconstruction_anchored_readout_20260920.json')
    out=Path('artifacts/native_reconstruction_readout_cv_transfer_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    parent=json.loads(rp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    rows={t:comparison(c,parent['source_selection'][t]) for t,c in parent['choices'].items()}
    changed={t:v for t,v in rows.items() if v['changed']}
    summary=dict(changed=len(changed),unchanged=len(rows)-len(changed),
        held_improving=sum(v['held_site_delta']<0 for v in changed.values()),
        held_worsening=sum(v['held_site_delta']>0 for v in changed.values()),
        mean_cv_delta=float(np.mean([v['cv_delta'] for v in changed.values()])),
        mean_held_delta=float(np.mean([v['held_site_delta'] for v in changed.values()])),
        rank_correlation=float(spearmanr([v['cv_delta'] for v in changed.values()],[v['held_site_delta'] for v in changed.values()]).statistic),
        both_group_improving=[t for t,v in changed.items() if all(d<0 for d in v['held_group_deltas'].values())])
    out.write_text(json.dumps(dict(targets=rows,summary=summary,bindings={str(p):sha256(p) for p in (Path(__file__),rp)},runtime_modified=False,promotion_allowed=False,
        limitation='Same group-equal/site-equal weighting in CV and held source. Descriptive selection optimism after choosing CV minimum; targets dependent. Cannot separate encoder in-sample reuse, model selection optimism, or distribution shift without an independent encoder-fit split. No new choice or thresholds.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

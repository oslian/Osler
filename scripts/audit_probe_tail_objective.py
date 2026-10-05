"""Source-fixed constant baselines and tail decomposition, not a new predictor."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.refit_anchored_body_readout import source_weights
from whole_body_jepa.multisource_intervals import sha256


def weighted_median(y,w):
    order=np.argsort(y);y=y[order];w=w[order]
    return float(y[np.searchsorted(np.cumsum(w),w.sum()/2,side='left')])


def source_statistics(y,w):
    w=w/w.sum();center=weighted_median(y,w);mean=float(w@y)
    distance=abs(y-center);cut=float(np.quantile(distance,.99));tail=distance>=cut
    loss=(y-mean)**2;total=float(w@loss)
    return dict(mean=mean,median=center,tail_threshold=cut,tail_weight=float(w[tail].sum()),
        tail_squared_loss_fraction=float(w[tail]@loss[tail]/total) if total>0 else 0.)


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_current_raw_body_probe_20260920.json')
    out=Path('artifacts/native_reconstruction_probe_tail_objective_20260920.json')
    if out.exists():raise ValueError('Existing report')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    mp=root/'site_balance_manifest_20260920.pt';manifest=torch.load(mp,weights_only=False)
    data={r:{t:[] for t in parent['targets']} for r in ('fit','encoder_tune')};digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        p=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt';digest.update((p.name+sha256(p)).encode());saved=torch.load(p,weights_only=False)
        if saved['shard']!=s:raise ValueError('Identity mismatch')
        for r in saved['rows']:data[s['role']][r['target']].append(dict(group=s['group'],truth=r['truth'],site=r['site']))
    if digest.hexdigest()!=parent['cache_digest']:raise ValueError('Cache changed')
    result={}
    for t,rows in data['fit'].items():
        y=np.array([r['truth'] for r in rows]);groups=np.array([r['group'] for r in rows]);sites=np.array([str(r['site']) for r in rows])
        stat=source_statistics(y,source_weights(groups,sites));per={}
        for g in ('selected','broad'):
            values=np.array([r['truth'] for r in data['encoder_tune'][t] if r['group']==g])
            per[g]=dict(patients=len(values),mean_constant_mae=float(abs(values-stat['mean']).mean()),
                median_constant_mae=float(abs(values-stat['median']).mean()),
                original_mae=parent['targets'][t][g]['mae']['original'],body_ridge_mae=parent['targets'][t][g]['mae']['body'])
        result[t]=dict(source=stat,held=per)
    summary={}
    for g in ('selected','broad'):
        delta={t:v['held'][g]['body_ridge_mae']-v['held'][g]['original_mae'] for t,v in result.items()}
        positive=sum(max(v,0.) for v in delta.values());rank=sorted(delta,key=delta.get,reverse=True)
        summary[g]=dict(top_regressions=rank[:2],top_two_fraction_of_positive_regression=sum(delta[t] for t in rank[:2])/positive,
            mean_constant_macro=float(np.mean([v['held'][g]['mean_constant_mae'] for v in result.values()])),
            median_constant_macro=float(np.mean([v['held'][g]['median_constant_mae'] for v in result.values()])))
    out.write_text(json.dumps(dict(targets=result,summary=summary,bindings={str(p):sha256(p) for p in (Path(__file__),pp,mp)},runtime_modified=False,promotion_allowed=False,
        limitation='Descriptive source99th-distance tail, no clipping/deletion. Constant mean versus median isolates loss preference only for constant estimators, not the ridge encoder causal mechanism. Top-regression decomposition is post-hoc, not target selection.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(json.dumps({t:result[t] for t in ('troponin_i','cpk','glucose','ptt')},indent=2))


if __name__=='__main__':main()

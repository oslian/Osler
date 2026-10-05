"""Source-only constant-predictor diagnostic of MAE versus SmoothL1."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scripts.train_joint_source_reconstruction import equal_group_median
from whole_body_jepa.multisource_intervals import sha256


def huber_center(groups):
    y=np.concatenate(groups)
    if y.min()==y.max():return float(y[0])
    def gradient(x):return float(np.mean([np.clip(x-g,-1,1).mean()for g in groups]))
    return float(brentq(gradient,float(y.min()),float(y.max())))


def main():
    out=Path('artifacts/native_reconstruction_joint_constant_objective_20260918.json')
    if out.exists():raise ValueError('Existing diagnostic')
    ip=Path('data/research_recovery/broad_source_20260917/joint_source_manifest.pt')
    rp=Path('artifacts/native_reconstruction_joint_source_manifest_20260917.json')
    report=json.loads(rp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Manifest changed')
    index=torch.load(ip,weights_only=False);variables=index['variables'];targets=index['targets']
    data={g:{r:{t:[]for t in targets}for r in ('fit','encoder_tune')}for g in ('selected','broad')}
    for item in index['shards']:
        p=Path(item['path'])
        if sha256(p)!=item['sha256']:raise ValueError('Shard changed')
        for c,e in torch.load(p,weights_only=False)['samples'].items():
            if variables[c]in targets:data[item['group']][item['role']][variables[c]].append(e['truth_normalized'])
    results={}
    for t in targets:
        fit=[np.array(data[g]['fit'][t])for g in data]
        l1=equal_group_median(fit);huber=huber_center(fit)
        rows={}
        for g in data:
            rows[g]={}
            for role in data[g]:
                y=np.array(data[g][role][t]);a=float(abs(y-l1).mean());b=float(abs(y-huber).mean())
                rows[g][role]=dict(patients=len(y),median_mae=a,huber_mae=b,huber_minus_median=b-a)
        if np.mean([r['fit']['huber_minus_median']for r in rows.values()]) < -1e-9:
            raise ValueError('Weighted median is not MAE optimal')
        results[t]=dict(median=l1,huber=huber,groups=rows)
    out.write_text(json.dumps(dict(targets=results,bindings={str(p):sha256(p)for p in (Path(__file__),ip,rp)},
        limitation='Constant model only. Source-selection descriptive; not neural conditional loss attribution or promotion.',
        runtime_modified=False),indent=2)+'\n')
    for g in data:
        print(g,'tune mean Huber-minus-median',np.mean([r['groups'][g]['encoder_tune']['huber_minus_median']for r in results.values()]))
    for t in ('glucose','lipase','cpk','troponin_i','temperature'):
        print(t,results[t])


if __name__=='__main__':main()

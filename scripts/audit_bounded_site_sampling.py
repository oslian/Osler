"""Measure source-only weight intervention without fitting or label access."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from scripts.bounded_site_sampling import bounded_weights
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    ip=root/'site_balance_manifest_20260920.pt'
    out=Path('artifacts/native_reconstruction_bounded_site_sampling_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    parent=json.loads(pp.read_text())
    if sha256(ip)!=parent['private_sha256']:raise ValueError('Source mapping changed')
    data=torch.load(ip,weights_only=False)
    variables=torch.load(root/'joint_source_manifest.pt',weights_only=False)['variables']
    pools=defaultdict(list)
    for (group,role,patient,c),site in data['sample_sites'].items():
        if role=='fit':pools[group,c].append(site)
    results={}
    for (group,c),sites in pools.items():
        w=bounded_weights(sites);counts=Counter(sites)
        old=np.array(list(counts.values()),float)/len(sites)
        new=np.array([w[np.asarray(sites)==s].sum()/len(sites) for s in counts])
        results.setdefault(group,{})[variables[c]]=dict(patients=len(sites),sites=len(counts),min_weight=float(w.min()),max_weight=float(w.max()),
            fraction_at_upper=float(np.mean(w>=2.-1e-10)),effective_patient_fraction=float(w.sum()**2/(w@w)/len(w)),
            old_effective_sites=float(1/(old@old)),new_effective_sites=float(1/(new@new)),total_variation=float(abs(new-old).sum()/2))
    out.write_text(json.dumps(dict(groups=results,lower=.5,upper=2.,label_access=False,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (pp,ip,Path(__file__),Path('scripts/bounded_site_sampling.py'))}),indent=2)+'\n')
    for g,targets in results.items():
        print(g,'median total variation',np.median([r['total_variation'] for r in targets.values()]),
              'minimum effective patient fraction',min(r['effective_patient_fraction'] for r in targets.values()))


if __name__=='__main__':main()

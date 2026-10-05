"""Outcome-blind source site support with patient-linked hospital components."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_reconstruction_site_support_20260917.json')
    if output.exists():
        raise ValueError('existing support evidence')
    root = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    qp, ip = root/'queries.parquet', root/'expanded_reconstruction_20260913/index.pt'
    catalog_path = Path('artifacts/native_reconstruction_expanded_20260913.json')
    catalog_report = json.loads(catalog_path.read_text())
    for path in (qp, ip):
        hashes = [h for p,h in catalog_report['bindings'].items() if Path(p).name==path.name]
        if hashes != [sha256(path)]:
            raise ValueError('changed cohort/index')
    q = pd.read_parquet(qp)
    if q[['uniquepid','hospitalid']].isna().any().any():
        raise ValueError('missing provenance')
    index = torch.load(ip,weights_only=False)
    hospitals = sorted(q.hospitalid.unique())
    positions = {h:i for i,h in enumerate(hospitals)}
    row, col = [], []
    patient_sites = q.groupby('uniquepid').hospitalid.unique()
    for sites in patient_sites:
        for a,b in combinations(sites,2):
            row.extend([positions[a],positions[b]])
            col.extend([positions[b],positions[a]])
    graph = csr_matrix((np.ones(len(row)),(row,col)),shape=(len(hospitals),len(hospitals)))
    n_components, labels = connected_components(graph,directed=False)
    patient_component = {}
    for patient, sites in patient_sites.items():
        components = {int(labels[positions[h]]) for h in sites}
        if len(components)!=1:
            raise ValueError('linked patient split between components')
        patient_component[patient] = components.pop()
    heldout_people = set(index['roles']['hospital'])
    heldout_components = {patient_component[p] for p in heldout_people}
    catalog = catalog_report['source_eligible']
    counts = {t:{} for t in catalog}
    quarantined = set()
    source_people = set()
    for shard in index['shards']:
        patient = shard['patient']
        if patient in heldout_people:
            continue
        component = patient_component[patient]
        if component in heldout_components:
            quarantined.add(patient)
            continue
        source_people.add(patient)
        path = durable(shard['path'])
        if sha256(path)!=shard['sha256']:
            raise ValueError('changed shard')
        s = torch.load(path,weights_only=False)
        if s['patient']!=patient:
            raise ValueError('shard identity mismatch')
        for c in s['samples']:
            target = index['variables'][c]
            if target in counts:
                counts[target][component] = counts[target].get(component,0)+1
    targets = {}
    for target, by_site in counts.items():
        values = sorted(by_site.values(),reverse=True)
        total = sum(values)
        supported = [n for n in values if n>=30]
        # Separate tune/test single components; all remaining components fit.
        feasible = sum(total-a-b>=30 for a,b in combinations(supported,2))
        targets[target] = dict(source_patients=total,component_patient_counts=values,
            components_with_30=len(supported),feasible_unordered_tune_test_pairs=feasible,
            largest_component_fraction=values[0]/total if total else None,
            single_component_tune_test_feasible=feasible>0)
    result = dict(complete=True,targets=targets,source_patients=len(source_people),
        hospital_components=n_components,heldout_components=len(heldout_components),
        quarantined_source_patients=len(quarantined),
        feasible_targets=[t for t,v in targets.items() if v['single_component_tune_test_feasible']],
        runtime_modified=False,promotion_allowed=False,
        limitation='Support only, no new partition selected or training. Feasible pairs target-specific, not a shared partition covering all targets. Multiple small components could jointly support a role even when this single-component test fails. Site-connected cohorts are not necessarily independent physiological populations.',
        bindings={str(p):sha256(p) for p in (Path(__file__),qp,ip,catalog_path)})
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('source',len(source_people),'components',n_components,'quarantined',len(quarantined))
    for t,v in targets.items():
        print(t,v['source_patients'],v['components_with_30'],v['feasible_unordered_tune_test_pairs'],v['component_patient_counts'][:5])


if __name__=='__main__':
    main()

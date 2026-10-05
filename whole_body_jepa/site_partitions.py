"""Patient-linked site components for history-safe development partitions."""
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def site_partition(provenance, reserved_patients):
    if provenance[['uniquepid', 'hospitalid']].isna().any().any():
        raise ValueError('complete patient/site provenance required')
    sites = sorted(provenance.hospitalid.unique())
    columns = {s: i for i, s in enumerate(sites)}
    patient_sites = provenance.groupby('uniquepid').hospitalid.unique()
    if not set(reserved_patients).issubset(patient_sites.index):
        raise ValueError('reserved patient absent')
    rows, cols = [], []
    for hospitals in patient_sites:
        for a, b in combinations(hospitals, 2):
            rows.extend([columns[a], columns[b]])
            cols.extend([columns[b], columns[a]])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sites), len(sites)))
    _, labels = connected_components(graph, directed=False)
    components = {}
    for patient, hospitals in patient_sites.items():
        ids = {int(labels[columns[h]]) for h in hospitals}
        if len(ids) != 1:
            raise ValueError('patient spans graph components')
        components[patient] = ids.pop()
    reserved = {components[p] for p in reserved_patients}
    groups = {}
    for patient, component in components.items():
        groups.setdefault(component, []).append(patient)
    source = {c: p for c, p in groups.items() if c not in reserved}
    total = sum(map(len, source.values()))
    if len(source) < 3:
        raise ValueError('at least three source components required')
    fractions = {'fit': .6, 'tune': .2, 'evaluation': .2}
    counts = dict.fromkeys(fractions, 0)
    assignments = {c: 'reserved' for c in reserved}
    # Largest components first, then largest remaining patient-count deficit.
    # This uses no target values, errors, or target-specific eligibility.
    for c in sorted(source, key=lambda c: (-len(source[c]), c)):
        role = max(fractions, key=lambda r: total*fractions[r]-counts[r])
        assignments[c] = role
        counts[role] += len(source[c])
    if not all(counts.values()):
        raise ValueError('empty source role')
    roles = {p: assignments[c] for p, c in components.items()}
    check = provenance.assign(role=provenance.uniquepid.map(roles))
    if (check.groupby('hospitalid').role.nunique() != 1).any():
        raise ValueError('hospital crosses partition boundary')
    return roles, components

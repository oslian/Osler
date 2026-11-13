"""Patient-only role assignment for frozen reconstruction shards."""


def patient_roles(queries, partition):
    required=('fit','tune','evaluation','hospital')
    mapping={}
    for role in required:
        if role not in partition:raise ValueError('missing partition role')
        for patient in set(queries.loc[partition[role],'uniquepid']):
            if patient in mapping:raise ValueError('patient occurs in multiple roles')
            mapping[patient]=role
    if set(mapping)!=set(queries.uniquepid):raise ValueError('partition misses patients')
    return mapping

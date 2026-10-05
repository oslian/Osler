"""Source-name evidence for an already accepted native glucose observation."""
import numpy as np


SOURCE_NAMES = ('bedside glucose', 'glucose')


def source_indicator(group, observed):
    """Unknown/ambiguous provenance stays zero; this is not an assay identity."""
    result = np.zeros(len(SOURCE_NAMES), dtype=np.float32)
    if not observed:
        return result
    labs = group.loc[group.source_kind.eq('lab_available') & group.variable.eq('glucose')]
    labs = labs.loc[labs.measured_at.notna()]
    if labs.empty:
        raise ValueError('observed glucose has no measured event')
    latest = labs.loc[labs.measured_at.eq(labs.measured_at.max())]
    names = latest.source_test_name
    if names.isna().any() or names.nunique() != 1:
        return result
    name = names.iloc[0]
    if name in SOURCE_NAMES:
        result[SOURCE_NAMES.index(name)] = 1.
    return result

# Fixed seed averaging diagnostic

Average all seven existing whole JEPA point forecasts equally; compare with
the same seven-seed own-target average, persistence and each whole member.
No weight fitting, subset selection or hospital-specific routing. Use only
the common hospital role with no fit/tune patient overlap in any member.
Do not average source evaluation predictions: other seeds may have fitted
those patients. Inherited pretraining overlap remains unverified.

Patient-equal MAE, paired bootstrap1000; all and bracket-after groups. Reused
hospital validation, not a new lockbox or seven independent replication tests.
Member comparisons descriptive and multiple, not promotion. No calibrator or
runtime replacement. This tests optimization variance mitigation under the
current rate-query task, not a new shared-state architecture.

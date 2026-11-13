# Bedside revision rejection check

Actual qualified cache:1,347,546rows and1,347,546distinct stay/source identities,
zero repeated identity versions.2,352quality-rejected rows remain present.
Consequently same-identity revision resurrection is not an observed explanation
of this bedside cohort's regression. This does not exclude different-source
conflicts or every raw-source quality issue.

Added a synthetic valid -> invalid same-identity revision regression: the first
query sees80, the second retains invalid0 evidence but emits no observation;
both snapshot and native replay reject the old version without changing the
earlier query. No production-code change was needed. No training, accuracy,
runtime or promotion change. Tests and local cache inventory are evidence;
this is not a newly repaired forecast cause.

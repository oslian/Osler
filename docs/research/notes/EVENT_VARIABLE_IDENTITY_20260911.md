# Variable identity ingestion check

Canonical deduplication previously checked time/value disagreement but not
variable identity. Synthetic same-ID/same-value/different-variable records
could be silently deduplicated; a later version could also change the analyte.
The canonical input now rejects variable changes within a stay/source ID,
and rejects missing variable when a multivariable column is supplied. Existing
single-variable replay tables have no variable column and remain valid.
Separate stays may reuse an ID; valid same-variable revisions remain supported.

Actual body-lab cache:646120 rows,646120 event IDs,0 cross-variable collisions;
all646120 rows accepted after the check. This rules out this collision pattern
as the current cache's failure cause; it is ingestion validation, NOT a forecast
repair.26 related tests pass. No model training, runtime weight or registry
change, intervals, or promotion. This raw-lab cache is not every local dataset.

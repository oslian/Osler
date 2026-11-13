# Current pretraining source identity check

Loaded the actual release checkpoint's training.event_examples path. Its
current file has580,980rows/3,291subjects/4,559stays. All stay-to-subject mappings
were verified against native eICU patient.csv, not inferred from string formats.
After normalizing the explicit eicu: stay namespace, intersections with this
native rate-query paired cohort are zero subjects, zero stays and zero hospitals.

Checkpoint metadata describes563,736rows/3,286subjects. This could reflect
training filtering or a changed file; neither is established by row counts.
There is no historical input hash or exact training membership in the payload.
Therefore this verifies CURRENT FILE independence, not historical pretraining
independence. It neither establishes leakage nor supports leakage as the
reason for the latest incremental failure. Current source hashes now recorded
in `artifacts/native_pretraining_provenance_20260911.json`.

Repeated use of the recent evaluation/hospital patients remains a separate
method-selection limitation even if pretraining independence is confirmed.
These are development diagnostics, not repeatedly fresh confirmatory tests.

Next training must record immutable source hashes and patient-role membership
privately, including any initialization pretraining, and evaluate a frozen
design on genuinely untouched patients/data before promotion. Do not invent
historical hashes or infer all-data training from a path alone. No weights,
registry, intervals or runtime changed. No accuracy or root-cause claim.

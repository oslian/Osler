# Ancestry Identity Crosswalk

The entire CURRENT ancestral event source contains4559stays,3291patients and
12hospitals. The reconstruction cohort contains1916stays and1763patients.
Both source subject/stay/hospital identities were verified against the original
eICU patient.csv.gz table. Query stay IDs use an explicit eicu: namespace;
normalization removes only that verified prefix. No guessed hash/string mapping.

There are ZERO patient intersections and ZERO stay intersections. Each nested
role also has ZERO hospital intersections with the current ancestral source:
fit11sites, encoder-selection4, readout-selection7, evaluation11, reserved21.

This resolves the namespace uncertainty in the previous warm-start audit. Zero
overlap is now supported by the original identity crosswalk, not merely direct
string mismatch. The previously missing crosswalk should not be repeated as
unresolved or used as a reason to assert demonstrated patient leakage.

Important boundary: the historical training row/subject count mismatch and
absent historical source hash remain. This verifies the entire current source
file is disjoint, not that every historical optimizer exposure is reconstructed.
Checkpoint ancestry is byte-matched separately. No exact historical leakage or
its effect on model performance is established.

## Decision

Do not replace the initializer solely on an unproven overlap suspicion. The
nested encoder/head selection repair can proceed as a warm-start DEVELOPMENT
experiment with the provenance caveat explicit, rather than claiming clean
historical independence or a new lockbox. Source-scaler provenance remains a
separate contract. Any from-scratch alternative needs a matched experiment, not
an assumption that warm start caused the failures.

Aggregate-only output, original identities stay private. No model/runtime or
promotion change. Artifact:
`artifacts/native_reconstruction_ancestry_identity_20260917.json`.

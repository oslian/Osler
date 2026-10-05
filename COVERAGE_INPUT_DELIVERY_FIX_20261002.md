# Input Delivery Gap: Confirmed and Repaired

## First Failure

The September30 GitHub source archive contained the strict weight loader but not
the native-history builder named in its usage guide. Weight loading therefore
passed while a downstream user could not convert their patient measurements to
the required tensor. Previous export parity started with a private cached native
tensor and missed this boundary. This was a packaging/integration failure, not
evidence of model inaccuracy or a user's incorrectly assembled input.

## Repair

- Added `whole_body_jepa.coverage_input`: canonical event CSV parsing, explicit
  units, availability-time filtering, original conflict/revision semantics,
  native grouping, target/sibling masking and fit-only normalization.
- Added `scripts.predict_coverage_csv`: checkpoint schema listing and complete
  CSV-to-current-estimate command, no private registry/cache required.
- Exporter now includes the actual history schedule, event tokenizer, builder,
  dependencies, synthetic CSV, tests and full bilingual usage guide.
- Missing values are not observed zeros. Future events, other stays, wrong units
  and treatment rows cannot silently enter. No independent history fails closed.
- Clinical source-table extraction/unit qualification is still upstream. The
  public interface is canonical events, not guessed MIMIC/eICU item mapping.

## Verification and Scope

In an extracted directory outside the repository, the documented CSV command
completed and four input-contract tests passed. All four actual exported folds
and all47 trained targets passed188 synthetic tensor/prediction parity checks;
omitting the requested target row leaves predictions identical. Project imports
were verified to reside inside the extracted archive. All asset hashes checked.

The original tokenizer stores redundant module-presence fields before masking;
the model recomputes presence after masking. Removing a target row can change
that redundant field while preserving the effective numeric/mask/age input and
prediction exactly. The verifier separately tests same-input tensor parity and
missing-target prediction invariance; it does not force a fictitious target row.

Four checkpoint SHA256 values exactly match the prior release. No retraining,
runtime promotion or new calibrated interval was performed. Synthetic packaging
checks are not prediction accuracy evidence. No real patient rows were uploaded.

Published private prerelease:
https://github.com/Chun0613-code/WholeBody-JEPA/releases/tag/coverage-reconstruction-20261002-input-complete

Use its attached `coverage_candidate_source.tar.gz`, not the automatic GitHub
source ZIP. See `COVERAGE_CANDIDATE_USAGE.md` and
`artifacts/coverage_input_bundle_verification_20261002.json`.

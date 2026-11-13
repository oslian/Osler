# Additional Domains: Full Masked-History Acceptance

## Confirmed Result

The complete frozen candidate population was replayed, without redraw or
outcome-dependent exclusion. Existing source-version conflict handling,
availability-time anchors, <=1h measurement age, >anchor-48h history, and
whole-history target/sibling masking were retained.

| Quantity | Count |
|---|---:|
| Proposed patients |73,812|
| Fixed patient-target anchors |1,810,110|
| Accepted anchors |1,800,214|
| Rejected: no fresh conflict-qualified anchor label |5,467|
| Rejected: no independent history after masking |4,429|
| Patients with at least one accepted anchor |73,670|
| Targets with accepted examples |47|
| Targets with >=30 people in >=6 components |46|

The 99.4533% fraction is DATA ACCEPTANCE, not prediction accuracy, interval
coverage, biological identifiability, or promotion. The exception to six
supported components is myoglobin (396 accepted examples, two components
with >=30 people). Population support does not guarantee role-specific support.

| Target | Accepted patient-target examples | Components with >=30 |
|---|---:|---:|
| Creatinine |69,883|30|
| BUN |69,847|30|
| Glucose |70,161|30|
| Albumin |51,432|29|
| Total protein |49,763|27|
| Heart rate |67,359|28|
| MAP |62,670|27|
| INR |47,195|28|
| TSH |10,683|19|
| Cortisol |1,358|11|

These are current-value reconstruction examples, not repeated future labels.
For MAP, 3,628 candidates fail the fresh/conflict-qualified label condition;
the audit has not yet decomposed that reason into measurement-method conflict,
revision or other causes. Do not assume invasive/noninvasive disagreement is
the sole cause or relax the label gate on that basis.

## Verification And Execution

All 105 completed decision-shard hashes and bound code/input hashes were
verified. Sorting on patient/target yields an EXACT match to all original
candidate identity/provenance/time/priority columns; no duplicates or lost
anchors. Status counts reproduce the aggregate report. Every accepted example
has positive independent-variable and independent-event-group counts.
The earlier 850-anchor pilot matched the original reconstruction builder
exactly; the full run did not redundantly re-encode all examples with that
reference. This distinction remains explicit in the machine-readable report.

The run took 48,905.8seconds (about13.6hours). A tool-session interruption did
not stop the original process. Its PID and exact command were verified, and
the SAME process was allowed to complete; no second replay was started.
The report's 20260921 filename identifies the starting experiment, completed
on20260922. All outputs stay factual and private at row level.

## What This Changes

Confirmed: the broad additional population has substantially greater
independent observed-history/domain support than the old bounded dataset.
This removes missing/empty histories as a blocker for building a broad-domain
reconstruction experiment under the present mask contract.

Not established: those histories contain enough information to predict the
hidden values, that a shared encoder can use it, or that residual calibration
will transfer. The small documented-insulin selected-cohort limitation remains;
these broad examples do not recreate that group. Sparse-marker current support
does not establish forecasting ability.

Next assign prospective component-disjoint fit/calibration/test roles, verify
each target's role support and fit-only vocabulary, then fit matched physiology
and process controls. No prior outcome-error-based domain selection. Preserve
the original selected-group unresolved result rather than silently replacing
it with a broader easier cohort. No weights, runtime, intervals, gates or
promotion were changed by this audit.

Report: `artifacts/native_reconstruction_additional_masked_history_full_20260921.json`.

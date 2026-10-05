# Additional Domains: Masked-History Replay Parity

## Implemented

`reconstruction_support` uses the existing source-version/conflict replay,
then vectorized equivalents of the existing event encoder's latest-draw,
quality, unique-value and <=1h age rules. A requested anchor must have a fresh
actual target label at that exact availability time. The target and every
`related_siblings` variable are removed throughout the >anchor-48h history.
No remaining observation means rejection, not a replacement anchor.

This is acceptance/support inspection, not a replacement physiology encoder
or a model accuracy experiment. No numeric scaling or learned weights are
needed to determine observation-mask support.

## Completed Real-Data Comparison

Before full population use, choose one patient from each of the 38 connected
components by a fixed patient hash, independently of values or reconstruction
errors. Retain ALL that person's frozen candidate anchors. Compare each audit
decision with the existing `reconstruction_examples` output, and compare the
number of independent variables and event groups for every accepted example.

| Quantity | Count |
|---|---:|
| Patients |38|
| Fixed candidate anchors |850|
| Exact reference acceptance comparisons |850|
| Accepted |847|
| Rejected: no fresh conflict-qualified anchor label |2|
| Rejected: no independent history after masking |1|
| Acceptance or support-count mismatches |0|

Elapsed processing time about37.4seconds. This includes reference construction,
not just the new audit; it is not a reliable full-run duration estimate.

Three test cases include eight seeded randomized event streams compared against
the reference, revised source values, later conflicts, invalid quality, stale
measurements, sibling exclusion and the 48h boundary. All pass. Pilot and
tests establish observed parity, not a mathematical proof for every possible
input or future transformation.

## Boundary And Next Execution

The complete population has 1,810,110 fixed anchors / 73,812 people. Its masked
history replay has NOT run yet. The 847/850 pilot acceptance fraction must NOT
be extrapolated to population support, rare-target support or model accuracy.
The full mode of `scripts.audit_additional_masked_history` processes the same
fixed anchors without redraw and retains every rejection in its denominator.

Next complete that replay, count accepted target/component support, and define
prospective source/calibration/test roles. Fit-only vocabulary support must
also be checked after splitting. The prior selected-cohort mismatch is not
resolved by this broad-cohort pilot. No clinical subtype, model training,
calibration, promotion or runtime change.

Evidence:
`artifacts/native_reconstruction_additional_masked_history_pilot_20260921.json`.
Patient-level decisions remain private and hash-bound.

# BUN Source-Fit Offset Transfer

Freeze bounded7. Fit MAE-optimal scalar offsets (median truth minus prediction)
on existing529selected/9664broad source-fit pairs, separately for full and
body-off. Apply BOTH sets to the same90selected/3086broad source-selection
pairs. Verify fitting/selection identity disjointness, dependency hashes and
original per-patient errors to1e-7. Selection has been reused, not a new lockbox.

| Offset fitted on | Full offset | Body-off offset |
|---|---:|---:|
| Selected fit |-.318660319|-.245210037|
| Broad fit |-.383224323|-.306765934|

All values normalized current-reconstruction units, not clinical BUN units.

| Selection group | Original full MAE | Selected-fit calibrated full | Broad-fit calibrated full |
|---|---:|---:|---:|
| Selected90 |1.105465044|1.129361892|1.143023336|
| Broad3086 |.893611644|.822632048|.820201675|

Broad full improvements7.94%/8.21%; paired descriptive95%delta intervals
[-.0814566,-.0602856]/[-.0861025,-.0605907]. Selected changes+.0238968/+.0375583
have intervals crossing zero. No offset chosen after selection results.

After calibrating BOTH arms, body-off minus full remains negative in broad:
-.0113895 CI[-.0161229,-.0065607] for selected-fit offsets;
-.0087294 CI[-.0134211,-.0039317] for broad-fit offsets.
Selected body-off is worse after either calibration: +.039977/+.043780,
both descriptive intervals positive. Thus a constant offset explains part,
but not all, of the broad body-dynamics penalty. Global dynamics deletion
remains unsupported. This is not an exact causal decomposition of MAE.

Signed full mean error is-.289288selected versus+.112064broad. The body's
net prediction contribution is+.196417selected/+.135106broad with standard
deviations.123899/.123550; it is not simply a constant branch contribution.
Mean signed error is NOT the MAE-optimal median offset, especially with tails;
do not equate mean-error sign with the optimal correction for every patient.

## Decision

There is a measurable readout/calibration component beyond integration step
size. Source-fit intercept correction transfers to broad selection but not
reliably to selected selection, and a residual dynamics tradeoff remains.
This does not prove the remaining difference is physiological rather than
sample selection or observation support. No conditional patient router is
validated here. Future repair must preserve both groups and other targets;
do not enable a source-label-conditioned correction from these outcomes.

Two median/invalid-input tests pass; audit exit0, diff check clean. No release
weights, runtime, uncertainty intervals or promotion changed. Private signed
predictions retained for further diagnosis without rerunning inference.
Evidence: artifacts/native_reconstruction_bun_offset_transfer_20260920.json.

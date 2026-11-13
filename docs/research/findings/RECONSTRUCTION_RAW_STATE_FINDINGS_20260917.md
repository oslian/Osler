# Raw Summary Versus Shared Body Readout

Completed all33supported targets on the same strict-site fit/tune/evaluation
patients and target/sibling-masked native histories. Six unsupported targets
retained. Original evaluation MAEs reproduce within1e-7. Full event/state
features cached privately; no encoder update. Scaler/ridge fit on fit only,
alpha or fit-median selected on tune only. No model parameters were selected
from evaluation outcomes. Data have been inspected repeatedly, so these remain
development results, not a fresh external lockbox.

## Results

Raw-summary ridge beats the body-state ridge on zero targets by nominal paired
95% intervals. This does not prove equivalence or optimal encoding, but provides
no positive evidence that this raw summary recovers information lost by state.
Raw features265 vs body64 are not equal-capacity representations.

Body readout improves original readout on four targets: bicarbonate, phosphate,
lipase and triglycerides. The latter two select the fit median on tune for ALL
three probe sources: raw, process and body. Their gains remove harmful model
movement, not demonstrate new physiology learning.

Two nonconstant body readouts improve original, population and raw-process:

| Target | Original MAE | Body ridge MAE | Paired change95% interval |
|---|---:|---:|---|
| Bicarbonate |0.976083|0.906803|[-0.121058,-0.018264]|
| Phosphate |1.030522|0.975366|[-0.103165,-0.005806]|

Errors are normalized current reconstruction, not future trajectories or
physical-unit errors. Counts174/138patients. Intervals are nominal, uncorrected
for multiple comparisons. These two are follow-up candidates, not promotion.

Body ridge also significantly harms ALT, bilirubin, BUN, platelets and troponin I
versus original. Raw ridge harms ten original targets and improves two only by
median fallback. A universal readout replacement would cause regressions.

## Decision

This comparison does not support a blanket encoder-discarded-information root
cause. Some state information can be read more effectively, but replacing every
head is not justified. Any integration requires source-selected per-target
original-versus-probe comparison and fresh independent confirmation, not
choosing these two heads because evaluation looked favorable.

Same input source does not imply full task equivalence: raw summaries omit
sequence ordering except latest/mean/lag/count; nonlinear raw encoders remain
outside this contrast. No proof all raw information is uninformative.

No runtime weights, registry, intervals or promotion changed. Evidence:
`artifacts/native_reconstruction_raw_state_20260917.json`.

# Known event sign across representations

Seven frozen synthetic-real checkpoints. Fixed Ridge(alpha1) probes trained
only on source-fit patients, with source patient-equal centering/scaling and
fit weights. Same injected last-known-event sign as training. No neural update
or probe selection. Aggregate: `artifacts/synthetic_latent_20260911.json`.

| Median sign correlation (rowwise descriptive) | Fit | New source | Hospital |
|---|---:|---:|---:|
| Belief state | .2695 | .1606 | .0961 |
| Shared context | .1162 | .0100 | -.0204 |
| Future target latent | .1184 | -.0229 | .0525 |

Patient-equal sign MSE at hospital:1.0601belief,1.0514context,.9981future;
a constantzero predictor on +/-1 signs has squared error1 (zero-sign unsupported
queries reduce that slightly). These are probes, not paired significance gates.
Correlations are rowwise and must not be described as patient-equal accuracy.

The known sign is not strongly linearly recoverable even from belief and
becomes weaker in shared context on new patients. This narrows the synthetic
failure to event representation/fusion learning, not only the final value
decoder. Finite linear probes cannot establish information absence or prove
nonlinear recoverability impossible. Different latent dimensions and fixed
regularization also limit layer comparisons.

Together with small in-fit treatment gains and limited synthetic uptake,
prioritize a controlled learning-path intervention before adding more output
heads. A successful repair must first recover the synthetic signal, then
independently improve real-label forecasts with unchanged gates. No clinical,
causal, runtime, calibration, or promotion conclusion follows from this probe.

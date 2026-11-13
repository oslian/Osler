# MAP Outcome Weights and Support Increment

## Diagnostic

Same frozen 21 tree predictions. First average absolute errors within each
patient/stay/future-event identity, then average events within each patient,
then paired patient bootstrap (20,000 draws). Original row-then-patient MAEs
reproduce. The alternative is a sensitivity analysis, not a replacement gate.
Conflicting truth under the same future identity raises an error.

## Results

| Role, event-equal | Own MAE | Full MAE | Support MAE |
|---|---:|---:|---:|
| Fit | 8.23194 | 7.92861 | 7.94863 |
| Tune | 8.87286 | 8.83632 | 8.84611 |
| New source patients | 8.69348 | 8.70936 | 8.72415 |
| Hospitals | 10.10688 | 10.21583 | 10.20583 |

Full-own gain7/7fit but0/7tune/source/hospital, as before. Full-support gain4/7fit
but0/7new patients. Full-persistence remains7/7each role. Therefore repeated
future-outcome scoring does not rescue full-context incremental significance.

Support-own also gains7/7fit but0/7tune/source/hospital. The training advantage
does not require other numerical values; there is also a smaller numerical
increment in fitting, but neither transfers stably. Support includes missingness
and ages which can reflect both physiology and workflow. This does NOT prove
the model learned only hospital practices, nor identify a causal decomposition
of the full-fit gain. Separate trained models are not additive components.

No fitting, source selection, runtime, calibration or promotion changes.
This does not test event-equal TRAINING. Evidence:
`artifacts/map_native_outcome_weights_20260912.json`.

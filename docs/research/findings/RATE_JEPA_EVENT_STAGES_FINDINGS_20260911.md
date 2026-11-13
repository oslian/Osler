# Where the trained event perturbation becomes small

Frozen seven real-event checkpoints, full scores reproduced. Remove numerical
event inputs while retaining known masks; inspect three returned representations.
RMS perturbation divided by across-row RMS coordinate variation, median seeds:

| Stage | Source | Hospital |
|---|---:|---:|
| Belief state | .008611 | .012412 |
| Shared context latent | .005176 | .007514 |
| Potassium future latent | .005368 | .007082 |

The event perturbation is already small in the earliest inspected belief
state. It is not a large state change erased only by the final value head.
The later context stage is somewhat less sensitive, but not disconnected.
These ratios are descriptive in the learned coordinate system, not a fraction
of biological information, causal influence, or explained variance. No
inference about the adequacy of the signal follows from their size alone.

This narrows the next mechanism check to event ingestion/filter learning versus
weak conditional information, rather than increasing decoder capacity. The
audit does not prove that increasing sensitivity would improve predictions.
No fitted repair or promotion; no runtime/calibration change.
Aggregate `artifacts/rate_jepa_event_stages_20260911.json`.

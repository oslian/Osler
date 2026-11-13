# Future-Label Reconstruction Candidate Findings

## Question

The joint JEPA receives target-specific actual event delays, but its value head
predicts a nominal horizon such as one hour. A label observed slightly before
or after that nominal point adds timing noise to the physiological loss.

This experiment asks whether a conservative reconstruction of a label at the
nominal time improves held-out raw-event forecasting.

## Data Contract

Two independent sepsis 1-hour datasets were rebuilt from the same eICU raw
event tables (31,298 stays, 451,009 transition rows):

- `eicu_time_raw_event_smoke`: strict raw-event labels only.
- `eicu_time_interpolated_smoke`: identical inputs, with optional additional
  labels for bracketed continuous vital signs.

An interpolated label is allowed only when:

1. the target is a high-frequency continuous vital;
2. two numerical raw events, both after the anchor, bracket the nominal time;
3. their interval is at most 15 minutes for the one-hour task;
4. no raw event exists exactly at the nominal target;
5. the record stores both raw bracket timestamps, its gap, and a reduced loss
   weight of 0.50.

Sparse or jump-prone targets remain raw-only: glucose, lactate, chemistry,
coagulation, hormones, troponin, bilirubin, and all other laboratories are
never interpolated by this candidate.

## Coverage and Audit

The interpolation candidate added tightly bracketed one-hour labels for:

- heart rate: 339,905
- MAP: 152,009
- oxygen saturation: 314,193
- respiratory rate: 302,858
- temperature: 38,494

`whole_body_future_label_quality_audit.py` verified every interpolated label:
both endpoints were post-anchor raw events, the bracket was within the allowed
width, and the reconstructed label was exactly at the nominal horizon. The
candidate audit passed all 18 sepsis target contracts.

## Common Raw-Event Evaluation

Models must not be compared on their own training-label variants. The new
`whole_body_label_reconstruction_audit.py` trains each candidate separately,
but scores both on the same patient-held-out strict raw-event labels.

This controlled smoke used 120 stays, one training epoch, seven patient splits,
and a neutralized measurement-process input. Mean held-out MAE was:

| Target | Raw-label training | Interpolated-label training | Result |
| --- | ---: | ---: | --- |
| Heart rate | 5.007 | 4.949 | Small improvement, still loses to persistence |
| MAP | 8.923 | 8.943 | Slightly worse |
| Oxygen saturation | 1.892 | 1.888 | Small improvement, still loses to persistence |
| Respiratory rate | 3.125 | 3.146 | Slightly worse |
| Temperature | 0.229 | 0.232 | Slightly worse |

No target passed all required patient-held-out checks. The candidate remains
`candidate_only` and is not added to the runtime router.

## Conclusion

Future timestamp jitter is a real but partial source of supervision noise.
Removing some of it does not reliably improve physiological forecasting. This
rules out label timing alone as the dominant bottleneck and preserves the
stronger conclusion: target-specific information content, individual response
heterogeneity, and sparse future observations remain limiting factors.

No causal or clinical claim is allowed.

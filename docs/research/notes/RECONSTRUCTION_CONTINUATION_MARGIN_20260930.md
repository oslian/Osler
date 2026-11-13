# Continuation: Control Drift And Persistent Component Harm

## Question And Scope

The completed epoch-8/10 assessment improved the base descriptive screen from
17 to 21 targets, but the process control changed too. This posthoc diagnostic
uses all 47 targets and unchanged component support to distinguish real-model
error changes, control error changes, and recurring component mean harm.
It reuses the existing component-influence method; it is not a new training
experiment, independent validation, or promotion gate. No groups are excluded.

The exact identity is:

`change in (real error - process error) = real error change - process error change`.

Crossed comparisons hold the process control at epoch 8 while evaluating real
epoch 10, and vice versa. These are component-only descriptive comparisons,
not replacement patient/component gates. The old population predictor is
verified identical, so recurring harm against it is not caused by control drift.

## Results

| Added base screen | Real component MAE change | Process change | New real vs old process component p |
| --- | ---: | ---: | ---: |
| Chloride | -0.075416 | -0.049225 | 0.0001134 |
| Fibrinogen | -3.055091 | -0.502531 | 0.0001955 |
| Platelets | -0.595456 | -0.388643 | 0.0001233 |
| PTT | -0.123919 | +0.050724 | 0.0019465 |
| Uric acid | -0.154441 | +0.048980 | 0.0012816 |

Negative changes mean lower error; units differ by target and must not be
averaged directly. The unchanged family threshold is 0.05/47 (~0.001064).
Chloride, fibrinogen and platelets retain the crossed component screen. PTT
and uric acid do not. Conversely, their OLD real model against the NEW process
control passes this component check (p=0.0004714 and 0.0007661). Thus their
process-screen status is control-sensitive, not sufficient evidence of improved
real physiology learning. This does not invalidate the original matched-budget
comparison or prove that the real models made no progress. Neither passed the
previous complete incremental epoch-10-versus-8 screen.

MAP provides a clearer arithmetic warning: real component MAE increases
0.182649, while process MAE increases 0.294165. Its relative process margin
improves even though its own error worsens. It remains unpromoted. ALT's
incremental result is not explained by a weaker control: its real error drops
0.575795 and its process error also drops 0.088981.

Among the 26 failed base-screen targets, 21 have at least one component with
30 or more patients whose mean error exceeds the SAME population predictor
at BOTH epochs. Seventeen have persistent supported mean harm versus the
respective process controls. These are recurring mean signs, not statistically
proven subgroup harms; epochs reuse patients, labels and related weights.
They cannot be counted as independent replications or a unique domain-shift
cause. The other five targets (hemoglobin, lactate, LDH, magnesium, pH) lack
this particular supported recurring-population-harm pattern; that is not proof
of equivalence, adequate power, or readiness for promotion.

## Decision

Do not judge continuation by the number of base-screen crossings alone.
Retain the existing direct comparison with the prior real model and report
control drift alongside it. No original thresholds or runtime rules change.
Continued training has heterogeneous effects; it is not a demonstrated global
repair. Some failures involve recurring supported mean harms, while others
remain primarily evidence-limited under this diagnostic.

Before choosing a further training change, check whether the recurring-harm
pattern is also identifiable in source-only held-out components. An outer-derived
list of harmed components must NOT become a training selector or an inference
route. Existing failed source correction and weighting experiments remain
negative evidence; this analysis does not justify repeating them unmodified.

## Reproducibility

- Run `python -m scripts.diagnose_continuation_margin` once on the bound reports.
- Aggregate: `artifacts/native_reconstruction_continuation_margin_20260930.json`.
- All current assessment dependency hashes and component support are checked.
- Five new tests cover control deterioration without model gain, retained small
  groups, support changes, invalid errors, and duplicate components; eight
  related regression tests also pass.
- No patient rows, model weights, runtime registry, or intervals changed.
- This remains masked-current reconstruction on reused development data.

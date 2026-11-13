# Inherited Cohort Restriction: Confirmed Design Mismatch

## Trace

`build_rate_anchored_potassium.py` restricts stays to prior paired potassium
queries and documented same-label insulin rate changes within360minutes, removes
conflicting/simultaneous ambiguous changes, and establishes the query stay list.
`audit_map_native_anchor_support.py` changes query times but retains this stay
universe. `build_native_reconstruction_expanded.py` expands target anchors within
those patient/stay roles, not to general eICU patients. The strict and nested
partitions inherit that restricted cohort.

## Original-table verification

Re-extracted compatible insulin rates with the original rate parser and exactly
reproduced the unambiguous rate-change criterion. Every old query stay matches:

| Role | Patients | Stays | Stays with qualifying rate changes |
|---|---:|---:|---:|
| Fit | 536 | 569 | 569 |
| Encoder selection | 90 | 99 | 99 |
| Readout confirmation | 89 | 93 | 93 |
| Evaluation | 178 | 193 | 193 |
| Reserved | 870 | 962 | 962 |
| Expanded confirmation | 3433 | 4131 | 125 |

Only118/3433expanded patients have a qualifying rate-change stay(3.44%).
189expanded stays have any parsed insulin rate, so absence of a qualifying
change is not equivalent to no treatment. Missing capture also remains possible.

## What is established

The fine-tuning/readout source cohort is selected for an insulin-rate/potassium
study. The new confirmation cohort intentionally lacked that inclusion rule.
Thus the last experiment changes both support and case selection; it cannot
isolate small-sample power as the only variable. Broad current-state claims need
development data drawn for that broad task, not an inherited treatment-triggered
patient list. This is a concrete design mismatch in this experiment lineage,
not proof all releases were trained on this restricted cohort.

## What is not established

No controlled experiment has isolated this selection difference as the sole
cause of BUN/bicarbonate/WBC deterioration. Extraction/sampling differences,
representation and head estimation remain competing contributors. The recorded
insulin change is retrospective over a stay: do not feed it as pre-anchor context,
route using future treatment, infer causality, or remove unhelpful patients.

## Repair requirement

Use the already inventoried non-outcome-selected fit/encoder-role patients to
rebuild development inputs with the same event contracts as expanded confirmation.
Keep all existing evaluation and reserved patients outside fitting. Compare the
old selected-cohort source against the broader source under fixed checkpoint/head
budgets; report both old and expanded confirmation cohorts separately. This is
not a request to retune on the inspected expanded confirmation labels. All39targets,
including missing source heads, must remain visible. Additional training is pending.

Evidence: `artifacts/native_reconstruction_cohort_selection_20260917.json` binds
raw rates, query/provenance sources, parser and cohort builders. Per-target
normalized truth quantiles and original MAEs are descriptive, not a causal test.
No runtime, weights, intervals or promotion changed.

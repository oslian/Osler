# Guided Expert Assignment Experiment

Status: fixed before fitting. This follows the free-mixture negative experiment;
runtime remains unchanged.

## Question

The free three-expert density produced distinct candidate values, but legal routing
was not stable enough for a full pass. Is the missing ingredient direct expert-
assignment supervision, or simply more experts?

## Design

- Use the identical frozen 64-dimensional shared JEPA context, target/sibling/history
  masking, source roles, patient anchors and three evaluation datasets from the
  preceding masked mixture experiment. If an OS temporary cohort has been cleared,
  a raw-data rebuild is accepted only when the source and NWICU file hashes reproduce
  exactly. MIMIC had a later urine-only file repair, so its rebuilt file must instead
  reproduce every target's selected patient count and all three frozen-reference MAEs
  to absolute tolerance `1e-6`; otherwise evaluation stops before reporting results.
- Fit `guided_3` and `guided_5`. Source-fit truth quantiles create equally supported
  low/middle/high (or five ordered) bins. During fitting only, the reader receives
  Laplace mixture NLL plus `0.25 * cross_entropy(gate_logits, truth_bin)`. At inference
  the gate sees only the legally masked shared context, never the truth or its bin.
- The three-expert head has 3,561 parameters and five-expert head 3,535, both within
  0.9% of the 3,565-parameter point control. No padding or unused parameters.
- Fixed seven seeds, AdamW, batch size, steps 32/128/512 and source-only selection are
  unchanged. Fit equal-architecture wrong-patient and matched-step placebos.
- Compare legal routing to uniform and three wrong-patient routing permutations.
  The component values/scales remain attached to the original patient; only routing
  probabilities are moved. Report outcome-aware oracle only as an impossible upper
  bound; it cannot select or promote anything.
- Full pass still requires population, wrong-patient, trained-placebo, seven patient
  split and every supported care-unit screen. An incremental expert pass also must
  beat the frozen 32-step and selected heads, capacity-matched point and single-density
  controls, free mixture, uniform routing and all routing shuffles.

MIMIC and NWICU have already been inspected and are replication/development sets, not
fresh lockboxes. No external tuning, automatic promotion, interval, future, causal,
clinical or next-measurement claim is allowed. The 28 movable / 13 persistence runtime
cells remain unchanged.

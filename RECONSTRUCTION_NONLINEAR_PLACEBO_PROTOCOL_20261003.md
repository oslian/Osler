# Matched Nonlinear Body-State Placebo

The completed nonlinear readout screen leaves a specific gap: its process
summary control has265 rather than64 inputs, and output-permutation diagnostics
of the original neural readout do not validate the new nonlinear readout.

Keep the original frozen encoder, all47 masked current-value targets, all five
folds, fit-only scaling, full cached histories and existing body64 HGB models.
For each target/fold, fit ONE new HGB with identical PARAMETERS, sample weights,
64 input columns and iteration budget. Reassign full body vectors among fitting
patients WITHIN provenance component only. Use seed20261003 + target index,
stable sorted patient identities, a random ordering followed by a one-step
cyclic shift. This preserves group feature/label marginals and prevents identity
matches for groups of size>=2. Singletons stay unchanged and are counted. It is
one fixed placebo realization, not an exhaustive permutation test or a causal
intervention. No patient/component identity is an input feature.

Do not change actual labels, fit populations, hyperparameters, real model or
encoder. Query inputs use their own real state for BOTH readouts. Query labels
are never arguments to fitting. Verify source and outer real-body predictions
exactly reproduce the completed nonlinear artifacts before interpreting a new
comparison. Retain all targets including unsupported outer fold-target cells.

Evaluate both complete source-selection and outer grids with the fixed heads.
All target/fold checks and results must finish before interpreting a partial
positive. No choosing targets or fitting from the already-inspected outer data.
No source-to-outer retuning. Both sets are reused development, not an untouched
external lockbox; fits overlap and source people repeat across eligible folds.

Primary comparison: body_probe minus trained_shuffle, patient and component
paired absolute-error differences, existing5000 paired bootstrap intervals and
one-sided t-test screen p<.05/47 with>=30patients and>=3components. Source losses
are averaged within patient before testing. Outer patients appear in one fold.
Also report native, population and process-summary means without inventing new
screens. Intersection with previous8x47 incremental/control screens is only a
diagnostic conjunction, not a new runtime gate or independent replication.

A positive result shows that correctly paired fit states help the chosen
readout on these data. It does not establish physiology-only representations:
within-component illness severity and observation practices may remain mixed.
The encoder was trained on real pairings, so this tests READOUT learning, not a
fully shuffled encoder-training null. One fixed shuffle cannot demonstrate
robustness across placebo seeds. No forecast/interval/clinical/causal promotion,
runtime change or private row publication is authorized by this diagnostic.

# Frozen Shared-State Distributional Expert Experiment

Status: protocol fixed before fitting. Research only; no runtime replacement.

## Question

Can a conditional distribution and legal expert assignment recover additional
current measurements from the SAME masked JEPA body state? This is not a future
forecast experiment, nor a test that there are three biological patient types.

## Fixed Design

- Freeze all release files and the existing 64-dimensional shared encoder.
- Reuse the preceding experiment's exact source fit/tune/held-hospital split,
  representative anchor per patient, target masking and fresh measured truth.
- Hide each target, its entire history and near-deterministic siblings. Only
  anchor-visible other measurements reach the shared encoder.
- Evaluate every supported numeric release target, not only previous near-misses.
- Fit three readers: 64-54-1 L1 point (3,565 parameters), 64-53-2 single Laplace
  density (3,553), 64-48-9 three-component Laplace mixture (3,561). Counts differ
  by less than 0.4%; no unused padding parameters. Density likelihood isolates
  distribution learning from simply adding parameters.
- Existing release scaler; no new per-database transformation. Initialization
  uses source-fit median / component locations at q10/q50/q90, with common robust
  scale. Component identities are not constrained or given clinical labels.
- Torch MixtureSameFamily + Categorical + Laplace negative log likelihood.
  Softplus scale with fixed 0.05 normalized floor; no target clipping.
- Seven existing seeds, AdamW lr=0.003, decay=0.001, batch=128, gradient clip=1.
  Snapshots at 32/128/512 steps. Select using source-tune half mean plus half
  worst supported care-unit MAE. No external model selection or hyperparameter tuning.
- Each seed has an independently trained, equal-architecture wrong-patient
  placebo, with independent source-tuned checkpoint selection. Also test the
  placebo at the candidate's selected step count.
- Point reader output: seven-seed mean. Density output: median of the pooled
  predictive distribution (7 or 21 components), not mean component locations.
- For mixtures compare legal routing with uniform routing and three deterministic
  wrong-patient routing permutations. Components stay with their original patient;
  ONLY routing weights move. No outcomes are used by legal routing.
- Outcome-aware oracle may choose the closest frozen component location OR retain
  the legal prediction. It is a diagnostic bound on that finite choice set, not a
  feasible model, and never enters selection, promotion, or runtime.

## Evaluation and Interpretation

Source held hospitals, then previously inspected MIMIC and NWICU. These are
replication/development evaluations, NOT untouched external lockboxes. Source
selection and weights are frozen before any external scoring. Foundation encoder
training may overlap these patients: only the new reader intervention is held out.

Use unchanged seven patient split, care-unit, population-prior, wrong-patient and
trained-placebo paired-bootstrap screens. Compare to frozen old readers and the
two capacity controls. Report q10/q90 tail errors, routing entropy/usage, and
oracle gaps. If oracle improves but legal routing does not, expert outputs offer
options but that alone does not prove that pre-anchor information can select them.

No automatic promotion, no conformal claims for learned densities, no future
prediction claims, no clinical/causal claims or next-measurement prediction.
The 28 movable / 13 persistence serving cells remain unchanged. Raw patient rows
stay outside the repository; only aggregate reports and candidate readers are saved.

## Established Pattern

- Bishop, Mixture Density Networks (1994):
  https://www.microsoft.com/en-us/research/publication/mixture-density-networks/
- PyTorch distributions (local 2.12.0 API inspected):
  https://docs.pytorch.org/docs/stable/distributions.html#mixturesamefamily

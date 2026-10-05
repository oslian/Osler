# Auxiliary supervision exposure

Replayed the exact Mg-primary real-arm patient sampler through each selected
epoch. For every auxiliary loss, each active target receives .25 divided by
the number of targets active in that batch; its active rows split that weight.
No new fitting or changes to the source-selected checkpoints.

Across the targets supported in all seven splits, median per-step loss
coefficients are approximately .00867-.01081 versus primary Mg coefficient1.
This is an objective coefficient, NOT measured gradient magnitude, parameter
influence, clinical importance or proof that a hundredfold increase is correct.
Gradient norms and shared-parameter interactions can differ by target.

Patient coverage at selected epochs is incomplete for some auxiliaries:
minimum across splits is82.3% for troponin,88.3% for lactate,88.6% for INR;
glucose/K99.3%. Coefficient-based effective patient fraction is about65% for
troponin and94.5% for glucose/K. This measures weight concentration, not
independent statistical sample size or real-patient data sufficiency.

The existing experiment tests Mg-primary adaptation with light auxiliary
supervision, not balanced full-body training. Its Mg whole-own failure remains
valid; that primary receives full supervision. These numbers cannot explain
that failure by themselves and do not overturn earlier balanced experiments.
They do prevent interpreting this particular run as failure of equally trained
all-target dynamics. A task-balanced counterpart would need matched compute,
patient/target support, source selection, and per-target retention evaluation.

No runtime/calibration/promotion. Aggregate:
`artifacts/magnesium_auxiliary_exposure_20260911.json`.

# Source patient influence diagnosis

Reproduced all21 saved own/whole comparisons and42 arm MAEs exactly (1e-12):
matched, L1 and event-balanced methods across seven source seeds. No retraining,
new external data, outcome-based routing or patient exclusion.

The implemented metric averages errors within each patient, then across
patients. Original training samples patients uniformly before queries. There
is no row-weighted scoring mismatch in these inspected paths. The event-balanced
intervention deliberately changes within-patient weights, not patient totals.

Original whole-vs-own patient benefit fraction ranges50.0%-59.0%. Top10% patients
by absolute error difference account for27.0%-37.8% of absolute differences.
Removing any single patient preserves the negative mean difference in all21
comparisons (all three methods). Thus one isolated patient does not explain the
mean advantage, but neither does the advantage extend reliably to all patients.
Leave-one-out sensitivity is not a general robustness or significance test.

Seven original splits contain500 patient appearances but only194 distinct tune
patients, with pairwise Jaccard .136-.271. Additional source patients are all
training-only; tune patients remain drawn from the original cohort. This is
explicit in the trainer, not new leakage. Existing tune CIs also use patients
that select checkpoints. They cannot be interpreted as independent promotion
evidence or seven independent replications. Current hospital/MIMIC evaluations
still show no stable whole-own benefit; this audit does not invalidate those.

Next discriminating design: patient-disjoint fit/tune/evaluation drawn from the
entire source population, frozen before training; identical own/whole examples
and capacity. Do not select epochs on the evaluation partition. Previously
seen source patients are not a new untouched external lockbox, and warm-start
pretraining overlap must remain disclosed. This repairs a limited diagnostic
design; it does not promise more accurate predictions or assert a unique cause.

Artifact: `artifacts/potassium_patient_influence_20260911.json`.
Command: `python -m scripts.audit_potassium_patient_influence`.
No runtime weights, intervals, registry or promotion changes.

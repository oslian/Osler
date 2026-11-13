# Source Hospital Increment Audit

Six V2 fits, all available catalog targets, source tune and evaluation only.
Saved prediction hashes and pooled MAE differences reproduce. No external
hospital predictions were used to choose a model, threshold or transformation.

## Provenance Correction

An initial diagnostic attempt rejected a patient mismatch before writing results.
The reconstruction example's query_position indexes a patient-local anchor table,
not the global queries file. This is an error in the new diagnostic attempt,
not evidence that the original training or prior scoring used wrong patients.
The corrected audit verifies a unique hospital across each patient's cohort
stays. Eight of1763 patients span hospitals and remain unassigned. They stay in
pooled errors but cannot contribute to site-specific claims. No arbitrary first
hospital, dropped patient or silent denominator change is used. Recovering their
exact selected stay remains unfinished.

## Results

- Bilirubin: each source evaluation split has only ONE site with at least30
  target patients. That site's numerical increment is favorable6/6, but it
  cannot establish broad site transport. Removing one known site reverses the
  pooled mean in one split (71); the other five remain favorable under all such
  removals. Not entirely a single-site artifact, nor well-supported site-wide
  invariance.
- O2 saturation:3-4 source sites meet30patients per split; direction is mixed.
  Split19 has unfavorable means in all three supported sites despite favorable
  pooled source error. Split59 has three unfavorable supported sites out of four.
  Pooled source benefit therefore conceals heterogeneity already inside source.
- Creatinine:3-4 supported sites per split, all favorable except one in split59.
  Removing any one known site preserves favorable pooled mean in all six fits.
  This is descriptive robustness, not significance or promotion.
- Triglycerides: only0-1 supported sites per split; pooled harm remains after
  any single known-site removal in31/43/59/71. Hospital calibration alone cannot
  repair the earlier source generalization failure.
- Glucose: supported-site directions mix in every split. There is no universal
  source-site numerical improvement hidden behind the aggregate.

## Interpretation And Next Repair Constraint

Source support is concentrated for several targets. Random patient splits can
repeat the same hospital patterns; they cannot substitute for site-excluded
training. Before claiming a domain-robust repair, a new source-only experiment
must reserve hospitals for selection/evaluation with adequate target patient
support, while retaining unsupported targets explicitly. Site identity belongs
in audit/partition logic, not the physiology encoder. The existing diagnostic
does not test such retraining, and does not justify changing promotion gates.

Fixed-prediction leave-one-site-out is NOT leave-one-site-out model retraining.
All means are descriptive, patients/sites overlap across splits, and the30patient
cutoff is support accounting rather than a significance test. The remaining
within-site numerical/process differences do not identify biological causes.

Machine evidence: `artifacts/native_reconstruction_source_hospitals_20260917.json`.
No training, weights, intervals, runtime or promotion changed.

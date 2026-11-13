# Source Support / Ionized Calcium Localization

Protocol: `ENCODER_SOURCE_SUPPORT_PROTOCOL_20260910.md`.
Final aggregate: `artifacts/encoder_source_support_20260910.json`.
Same six repaired whole-body checkpoints, no refitting, same scored patients.
No serving/registry/calibration changes or promotion. Goal remains open.

## Fixed-Weight Results

Patient-equal glucose6h MAE, median of three source seeds; mg/dL.

| Candidate / input | eICU hospital | MIMIC |
|---|---:|---:|
| Whole unbounded / full | 38.339 | 35.461 |
| Whole unbounded / source-seen only | 38.339 | 33.323 |
| Whole unbounded / support >=20 patients | 38.337 | 33.187 |
| Whole unbounded-balanced / full | 38.485 | 34.105 |
| Whole unbounded-balanced / source-seen only | 38.485 | 33.324 |
| Whole unbounded-balanced / support >=20 patients | 38.486 | 33.259 |

Both restrictions improve both candidate modes on MIMIC in3/3 patient-bootstrap
CIs. eICU source-seen predictions are unchanged; support20 has no significant
eICU benefit. All variants produced zero negative predictions on these cohorts,
which does not guarantee range safety on unseen inputs. The three seeds reuse
the same validation patients, not independent databases; no multiplicity or
complete promotion claim. The simpler source-seen rule is retained as candidate
input protection, not a MIMIC-selected individual-variable policy.

## Localized Chain

1. Existing unit audit recorded630 eICU ionized-calcium events with system unit
   mg/dL, versus9950 raw MIMIC events in mmol/L (9946 retained after filtering).
   Availability whitelist accepts only mmol/L for this variable.
2. All three source fits therefore see zero ionized-calcium observations.
3. Existing checkpoint scaler has median3.7999999523 and scale0.1779121906.
   Accepted MIMIC ionized-calcium q05/median/q95 are0.95/1.12/1.28 mmol/L;
   normalized values are approximately-16.02/-15.06/-14.16.
4. Fixed-weight source-seen intervention changes no eICU input variables and
   ONLY ionized_calcium's value/mask/age on MIMIC, with derived organ presence.
   The artifact records this for every seed. Removing that unsupported channel
   consistently reduces error. This directly implicates that input channel's
   handling in these checkpoints; it is not merely an observational association.

The evidence supports an input/scaler/support contract defect. It does not prove
the precise origin of the OLD scaler's units; its original training lineage is
not reconstructed. Do not claim the body has no useful calcium-glucose coupling.

## Why Not Blindly Convert All eICU Calcium

An additional aggregate raw-unit inspection on the same eICU stays found both
matching mg/dL labels and conflicting system/interface labels. Of630 records,
182 have matching mg/dL labels ignoring case;448 have differing labels. Some
numeric/text values also differ. Numeric magnitude alone cannot establish which
label or transformation is correct. No guessed conversion was introduced.

[Official eICU lab documentation](https://eicu.mit.edu/eicutables/lab/) distinguishes
system and interface measurement names and numeric/text results; it does not
establish a blanket conflict-resolution rule for these records. Original input
provenance must be retained for a future unit-recovery experiment.

## Candidate Repair / Boundaries

`source_patient_support` counts distinct source-fit patients across legal lag
observations, not repeated rows or validation patients. `restrict_input_support`
sets unsupported values to normalized missing zero, mask0, unknown age, and
recomputes organ presence throughout history and current input. Patient identity,
time and supported channels remain unchanged. Two regression tests cover this.

Six local candidate checkpoints under
`/private/tmp/encoder_supported_weights_20260910` retain identical fitted weights
and add ordered variables, allowed_variables, source_patient_support and
input_policy=source_seen_only. Hashes and original weight hashes are in JSON.
To reproduce: load the matching base architecture, restore the checkpoint's
identity activation and state_dict; check ordered variables; apply
restrict_input_support using allowed_variables before inference. Do not load
these weights as a default tanh model or reuse the old conformal calibrator.

Still unresolved: source-seen whole remains worse than own unbounded median
MIMIC32.070. Source support protection is a verified partial repair, not proof
of transferable incremental whole-body dynamics. Seven-seed, care-unit/time,
calibration, retention and a pristine lockbox remain outstanding. Recovering
unit-verified source calcium and retraining is different from this fixed-weight
masking audit and has not been done.

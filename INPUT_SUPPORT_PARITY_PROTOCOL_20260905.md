# Source Input-Support Parity: Frozen Diagnostic

## Observed Defects in the Experiment Contract

The source transition file is AKI-selected, with 14 physiological input
variables. The MIMIC target file is whole-ICU, with a wider measured panel.
The readout audit fills unavailable source columns with NaN, then sends the
whole target panel into the same encoder. The vector length is fixed, but
the readout's training-time input support is not enforced at transfer time.
These are confirmed design differences, not proof of their causal contribution
to MAE. Do not call the two populations clinically matched.

## Paired Intervention

Derive the set of supported variables from eICU TRAIN-role inputs only;
never use labels or target-database outcomes to choose the set. A variable is
supported if it has at least one finite observation within the release age
limit. Verify excluded columns have no eligible observation anywhere in the
source file, so masking is source-equivalent and existing source-calibration
parameters remain applicable without refitting.

On MIMIC, set unsupported variable values and ages to missing for the complete
history BEFORE encoding. Preserve the same 53-position contract, all observed
zero values, raw target/current values, future labels, selected patients,
pre-anchor treatment context and frozen model/readout weights.

Compare original versus input-aligned fixed expert, binary selector and
conditional amplitude readout. Use the amplitude audit's unchanged controls,
conformal scales and evaluation policy; add a paired comparison to the original
input path. No cutoff, label-policy or weight tuning. This is an input-contract
intervention, not an extra model-head search.

Check exact reproduction of the preceding MIMIC point results first. Then
report all five 6h targets, coverage, CRPS, patient/unit/relative-time gates.
Do not promote automatically. Existing source-role gates already failed;
passing a MIMIC comparison cannot erase those failures.

## Remaining Limitations

The AKI-versus-whole-ICU selection difference is NOT fixed by masking inputs.
Repeatedly inspected MIMIC remains transfer validation, not an untouched lockbox.
The pretrained encoder may have seen a broader panel; masking is scoped to
these AKI-trained readouts, not a claim that the base encoder cannot use other
variables. Unsupported observations remain available to the rest of the system.
This test is allowed to falsify the input-support explanation. It must not
turn another negative result into proof that all data or model options are exhausted.

# Encoder Cross-Fit Feasibility Audit

This is a completed partition/support audit, not completed encoder training,
accuracy improvement or promotion. All 33 current reconstruction targets are
retained. No patient values, outcomes or prior model errors choose the folds.

## Verified Inputs And Partition

Use only the former source-fit population: 10,737 unique patients. Keep the
previous encoder-selection population outside this experiment. Verify the
source manifest and all bound dependencies. Resolve every patient's complete
hospital history against the raw eICU patient table, not just the hospital of
the selected target measurement. Every sample's site agrees with that history.

There are 14 hospitals but only **8 connected hospital components** because
44 patients have records at multiple hospitals. Patients connecting hospitals
must remain together in any joint encoder fold. Five outer GroupKFold splits
are assigned by these components. Each patient appears in outer test exactly
once. An inner component split separates fitting from checkpoint selection.
All assignments are stored privately with hashed provenance; only aggregate
support is published.

| Outer fold | Fit patients | Selection patients | Test patients | Selection sites | Test sites |
|---|---:|---:|---:|---:|---:|
| 0 | 4969 | 2796 | 2972 | 1 | 2 |
| 1 | 4969 | 2972 | 2796 | 2 | 1 |
| 2 | 5513 | 2972 | 2252 | 2 | 5 |
| 3 | 6299 | 2972 | 1466 | 2 | 3 |
| 4 | 6514 | 2972 | 1251 | 2 | 3 |

31/33 targets have at least 30 patients in every role of every fold. This is
patient support only, NOT adequate multi-hospital confirmation. Fibrinogen
has only 24 test patients in each of the last two folds. Troponin I has zero
selection examples in fold 0 and zero test examples in fold 1. They remain
explicitly unsupported in those roles, not omitted or recorded as failures.

## What This Changes

The proposed simple five-fold encoder experiment cannot silently use a single
inner selection split as strong multi-hospital evidence. Thousands of patients
do not imply many independent domains: its inner selection covers only one or
two hospitals. This is a concrete limitation of this proposed design, not proof
that previous patient-safe partitions leaked or that it caused every failure.

A legitimate subsequent comparison must either fix training duration before
outer scoring, without selecting epochs on these thin inner splits, or perform
inner component cross-fitting with all preprocessing and encoder learning
repeated. Unsupported targets must remain in the coverage report. Patient and
site-level results must both be reported; fold-trained models are dependent
through overlapping training sets and cannot count as independent replicates.

The current bounded7 encoder and its body caches cannot be used as independent
outer representations: they already saw these source-fit patients. Release
scaler and warm-start ancestry must separately be handled. The earlier raw
crosswalk found no overlap with the current ancestral file, but did not recover
every historical optimization exposure. Do not turn that caveat into a claim
of demonstrated contamination. A from-scratch comparator requires matched
training, not an assumption that warm start caused failure.

## Verification And Boundaries

Three tests cover transitive patient/site bridges, deterministic label-free
roles, exactly-once outer coverage and fail-closed insufficient components.
The full data preflight completes. No runtime weights, intervals or gates
changed. These are reused eICU development data, not an untouched external
lockbox. This audit concerns masked current reconstruction only.

Evidence: `artifacts/native_reconstruction_encoder_crossfit_preflight_20260920.json`.

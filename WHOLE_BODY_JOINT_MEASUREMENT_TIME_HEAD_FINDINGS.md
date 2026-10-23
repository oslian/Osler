# Whole-Body Joint JEPA: Measurement-Time Head Findings

## Scope

This experiment tests whether the model can learn **when a variable is likely to
be measured next**, separately from predicting the patient's physiology.

- Data: `eicu_time_augmented`
- Modules: sepsis, AKI
- Alignment: as-of, maximum source age 6 hours, horizon-scaled age policy
- Smoke cohort: 40 stays, 1,237 rows, 38 patients, 31 hospitals
- Evaluation: 7 patient-held-out seeds plus hospital-held-out evaluation
- Training: 1 epoch, batch size 32
- Status: research candidate only

## Implementation

The model now has an optional target-specific `next_measurement_time` head.
Labels are rebuilt from timestamped raw measurement events and are kept separate
from future physiological values. The head predicts `log1p(hours)` with a
positive transform.

The measurement-process loss is detached from the physiological latent. This is
intentional: hospital measurement behavior must not distort the representation
used for physiological forecasting. The head is an observation-process model,
not a causal or physiological intervention model.

The gate and scoring audit both expose the same CLI controls:

```text
--measurement-time-head
--measurement-time-weight 0.10
```

## Result

The head runs end to end and produces valid non-negative delay predictions. In
the small 7-seed audit, it did **not** produce a universal improvement over the
training-median time baseline.

The strongest preliminary signals were:

- `bilirubin`: mean patient-equalized log-time delta `-0.1306`; 2 split-level
  bootstrap checks passed.
- `lactate`: mean delta `-0.0779`; 6/7 splits improved; 1 strict bootstrap
  check passed.

Most other variables were mixed or worse than the baseline. Fast physiological
forecast behavior was not promoted based on this experiment; the measurement
head is detached and remains an independent observation-process task.

## Interpretation

This is evidence that timestamped measurement behavior contains target-specific
signal, especially for some sparse laboratory variables. It is **not** evidence
that a universal measurement-time model has been validated, and it does not
make sparse future physiology predictable by itself.

The correct product contract is therefore:

1. physiological forecast and next-measurement-time forecast are separate fields;
2. measurement-time predictions cannot feed back into physiological values;
3. each target requires its own patient, hospital, and conformal gates;
4. failed targets remain baseline/fallback and are not promoted;
5. no causal or clinical claim is allowed.

## Remaining validation

The smoke result is not a promotion result. A full evaluation must rerun the
same audit on the complete supported cohort and all registered modules, with
adequate observations per target. The candidate can only be promoted for a
target when the target-specific time-process error beats its baseline across
the required patient and hospital gates and has calibrated uncertainty.


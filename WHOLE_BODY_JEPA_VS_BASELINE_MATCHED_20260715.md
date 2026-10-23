# True JEPA vs Shared-Latent Baseline: Matched Gate Result

Date: 2026-07-15

## Design

The baseline and true JEPA used the same:

- 15 body-system modules;
- 65,077 aligned rows and 917 subjects;
- six horizons;
- two forecast epochs, batch size 2,048;
- cross-forecast weight 0.5;
- masking probabilities;
- seven patient-heldout seeds;
- hospital-heldout split;
- conformal and cross-system gates.

The only intended model difference was the future-latent JEPA objective:

```text
baseline: context -> numerical decoder loss
JEPA:     context -> predictor -> future latent
          future state -> EMA target encoder
          plus the same numerical decoder losses
```

## Results

| Cell | Baseline value pass | JEPA value pass | Baseline cross pass | JEPA cross pass | Baseline median delta | JEPA median delta |
|---|---:|---:|---:|---:|---:|---:|
| glucose@6h | 0/7 | 2/7 | 3/7 | 3/7 | +1.902 | +3.215 |
| potassium@6h | 2/7 | 2/7 | 4/7 | 4/7 | +0.00321 | +0.00312 |
| bicarbonate@12h | 2/7 | 2/7 | 5/7 | 2/7 | +0.0561 | +0.0466 |
| MAP@1h | 0/7 | 0/7 | 6/7 | 2/7 | +1.886 | +2.318 |
| creatinine@24h | 3/7 | 3/7 | 4/7 | 1/7 | +0.00649 | +0.00302 |

Both runs finished as `candidate_only` with zero fully validated cells.

## Conclusion

The true JEPA implementation is real and trainable, but this matched audit
does **not** prove that it is better than the existing baseline. On this
cohort and budget it is worse on glucose and MAP and less stable on the
cross-system ablation for bicarbonate and creatinine. It is therefore not
promoted and cannot replace the current target-gated factual router.

The result is a useful rejection of a stronger claim:

> A future-latent JEPA objective alone is sufficient to improve whole-body
> factual forecasting.

That claim is false for the current data, horizon contract, and training
budget. The remaining problem is not whether the JEPA code exists; it is how
to construct a target latent that is sufficiently dense, physiologically
stable, and comparable across asynchronous body-system measurements.

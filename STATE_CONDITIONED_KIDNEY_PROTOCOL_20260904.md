# State-Conditioned Kidney Forecast Protocol

## Question

Can one frozen shared whole-body JEPA state decide when persistence is the
correct stable-state expert and when a target-specific dynamic expert should
move, while improving both point error and predictive distributions?

## Frozen design

- Development and model fitting: eICU AKI cohort only.
- Transfer evaluation: MIMIC-IV; previously inspected, not an untouched lockbox.
- Targets: creatinine, BUN, urine output, potassium and bicarbonate at 6 hours.
- `P(dynamic)` reads only the shared context and patient belief blocks.
- Signed delta quantiles read the same legal 311-column frozen JEPA contract.
- Stable expert is persistence; dynamic expert predicts signed 5/50/95% delta.
- Controls are persistence, a plain equal-input residual head, and an exact-capacity
  shuffled-patient state model.
- Only strict/exact, positive-weight, post-anchor measured future pairs are used.
- Measurement-time prediction remains a separate task and is not trained here.

## Gates

Each target must pass eICU patient, eICU held-out-hospital and MIMIC transfer:

1. seven patient-split paired bootstrap comparisons;
2. care-unit and late-stay comparisons;
3. lower MAE than persistence, plain head and shuffled-state placebo;
4. lower CRPS than an empirical persistence distribution;
5. 90% interval row, patient-equal and care-unit-equal coverage in [0.87, 0.93].

All settings and cutoffs are selected without MIMIC labels. Failure leaves the
current release and persistence fallback unchanged. This is factual research,
not a causal or clinical-decision model.

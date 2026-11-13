# Patient-Equal Correction Contrast

Fixed before fitting: same point-only Huber correction, alpha0.1/epsilon1.35,
same support, source calibration folds and leave-component-out queries. Replace
component-equal weights with patient-equal weights summing to1 in BOTH scaler
and estimator. No parameter search, outer access, patient deletion or changed
scoring. Compare against unchanged and the saved component-equal point-only arm.
Report patient-weighted and component-equal errors and small-component harms.
Convergence continuation follows the same300+2700 rule. This diagnoses correction
training, not encoder learning; descriptive results cannot authorize promotion.

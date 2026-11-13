# Frozen event-summary change decomposition

Reproduce the seven event-window source evaluation scores with frozen models.
No fitting or external reads. Define a retrospective large-change subgroup
using each fitting partition's patient-weighted75th percentile absolute delta;
the complement is smaller-change, NOT a diagnosis of clinical stability.
Thresholds do not use evaluation truth; subgroup membership necessarily does,
and must never become a prospective routing feature.

For full/own/support and their original controls report patient-equal MAE,
paired CIs, delta correlation, signed bias, movement magnitude and direction
agreement on nonzero actual changes. Report MAE contributions with the original
pooled patient weights so the subgroup contributions sum to the pooled delta;
conditional subgroup MAEs use separate patient equalization and need not sum.

This tests whether the new input's small overall gain conceals opposing subgroup
effects. No deployment policy, amplitude rescaling, retrospective selector,
calibration, promotion, or claim about predictable risk follows.

Numerical correction: treat isclose(rtol=1e-8,atol=1e-8) to the source quantile
as equality, excluding tied values from strict-greater large-change membership.
This avoids floating subtraction around quantized potassium differences. V1
is preserved as a diagnostic; v2 recomputes all groups without changing models,
pooled scores, source quantiles, or evaluation patients.

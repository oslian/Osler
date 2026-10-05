# Glucose native rate-query diagnostic

Same5,526 original potassium-eligible query universe and seven patient roles;
score/train only5,438observed glucose pairs. This remains a potassium-selected
cohort, not all glucose patients. Original native clipped clock,48h events,
32epochs/16steps/batch32/patient-equal sampling/source tune selection/EMA loss.
Use original lr1e-4, not the synthetic-only successful but real-K-harmful rate.

Fit own glucose-only history without treatment, whole lab+real event values,
and whole lab+support only. Same backbone capacity, zero-expanded projection.
Own updates only on observed glucose; real/support have same lab/rate masks.
Source lab eligibility and event scaling use only eligible fit patients.
No target identity or future-measurement time feature. Primary real-support,
real-own, real-persistence and own-persistence patient bootstrap. No hyperparameter
selection on held-out hospital. Independent target test, not label relabeling
to overturn the potassium failure. No calibration, new lockbox, retention,
clinical/causal or runtime promotion. Pretraining overlap remains unverified.

# Recorded Pressor History Increment

Seven source splits, same expanded MAP fit and old source tune. Current-rate
tree baseline versus additional historical support versus additional historical
rate/delta plus support. Fixed1h/3h past cutoffs; same model family, width and
source-only leaf selection. Unsupported numeric columns masked using fit only.
All seven current-rate controls reproduce prior predictions exactly.

Median patient-equal MAE, mmHg:

| Scope | Current rate | History support | History rate + support |
|---|---:|---:|---:|
| Old source evaluation | 9.13632 | 9.14441 | 9.12633 |
| All source evaluation | 8.89208 | 8.85309 | 8.89388 |
| Old held-out hospitals | 10.39982 | 10.48818 | 10.40957 |
| All held-out hospitals | 10.14075 | 10.23638 | 10.14504 |

Common20k patient bootstrap: history rate versus current rate0/7gain in all
scopes;1/7harm on old source and old hospital, none on all rows. Versus history
support,1/7hospitalall gain,1/7sourceall harm, otherwise no significant change.
No stable repair. Raw physiologic/treatment history remains a non-neural
diagnostic here, not a new shared JEPA or complete event-sequence trial.

Recordedtime remains unverified availability. No administration duration,
start/stop inference, causal effect or online promotion claim. No conformal
audit or new external database test. Runtime weights/registry/calibrators
unchanged. Runner gained an explicit history experiment; old runner hashes are
historical, not current-code certification. Controls were actually rerun.

Evidence: `artifacts/map_pressor_history_training_20260912.json`;
25direct bindings verified. Private predictions remain local.

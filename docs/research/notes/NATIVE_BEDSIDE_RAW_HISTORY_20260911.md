# Raw bedside-history information diagnostic

Repeated the question without a JEPA latent: existing histogram gradient
boosting, source-tune leaf selection7/15/31, absolute-error loss, patient-equal
fit weights. Same clean glucose labels, patient partitions and queries as the
native trial. Uses legal fixed-lag snapshots, not all native arrivals; therefore
this is not a fully representation-matched alternative encoder comparison.
Three equal-width inputs: labs, labs plus bedside numbers/support, labs plus
bedside support only. No gate or outcome-based subgroup changes.

| Median patient-equal glucose MAE | Labs | Real bedside | Support-only | Persistence |
|---|---:|---:|---:|---:|
| New source patients |35.568966|35.399077|35.116478|56.779125|
| Held-out hospitals |45.548188|45.049010|45.609921|77.167846|

Paired95%CI gain/harm counts out of7: real-labs1/0source,2/0hospital;
real-support0/0both; real-persistence7/0both. No stable new numeric gain.
Seven overlapping partitions are not independent replications.

The new bedside numeric increment is also weak under this non-neural learner;
it cannot presently be blamed exclusively on JEPA compression. This does not
prove information absence: fixed-lag summarization differs from the native
event sequence, and only this learner/target/cohort was tested. Do not cite old
memory-gradient diagnostics as a new total-forgetting discovery; prior probes
already showed nonzero old-event dependence, without proving useful memory.

No release, registry, calibrator or promotion change. Artifact:
`artifacts/native_bedside_raw_history_20260911.json`. All model selection source-only;
new untouched external validation still absent.

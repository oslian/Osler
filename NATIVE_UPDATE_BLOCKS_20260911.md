# Native update block restoration

Seven original whole-glucose checkpoints; replace one parameter block at a time
with its exact initialization. Identical source-evaluation patients/labels,
original all-target predictions reproduced before intervention. No hospital
evaluation, refitting, model selection, interval or runtime change.

| Restored block | Glucose gain/harm | Mg gain/harm | Phosphate gain/harm | Calcium gain/harm | BUN gain/harm |
|---|---:|---:|---:|---:|---:|
| History filter | 0/3 | 4/1 | 5/1 | 2/4 | 1/5 |
| Variable/module mixers + state encoder | 0/1 | 7/0 | 5/0 | 5/2 | 1/5 |
| Shared future predictor | 0/0 | 7/0 | 6/0 | 7/0 | 0/6 |
| Three glucose-private modules | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

Counts are significant paired patient95% CIs across seven overlapping seeds.
Restoring shared_future reduces median source glucose MAE34.8408 to34.5144,
but not significantly in any seed. BUN worsens6/7, so this is not a universal
safe rollback. Creatinine improves3/7 under shared_future restoration.

The shared future decoder, not only the body encoder, is implicated in the
multi-target drift: restoring it changes mineral errors consistently while
glucose-private restoration leaves other outputs unchanged. Input-history and
mixing updates have different tradeoffs. These blocks co-adapted; mixed weights
are off the trained trajectory, so this is sensitivity localization, not proof
that training with this block frozen will yield these scores or that the block
is the sole biological/generalization cause.

The next controlled allocation experiment can keep shared_future fixed while
allowing history/context and target-private adaptation, with matched own input
and original controls. Unlike total freezing, this preserves encoder adaptation;
unlike uniform retention, it does not imitate every initial output. It must
still evaluate BUN and every supported target, since this diagnostic warns of
real costs. No repair/promotion is established yet. Base-model scope and reused
source data limitations remain; no new external lockbox or causal claims.

Artifact: `artifacts/native_update_blocks_20260911.json`.
Runner: `python -m scripts.audit_native_update_blocks`.

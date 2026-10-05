# Native event history matched training

Question: does a sequence of actual recorded events improve potassium forecasts
over ten history snapshots on the same 5,526 query/label pairs?

Freeze the seven existing patient partitions, source tune selection, 32 epochs,
16 steps/epoch, batch32 patient-equal sampling, AdamW and value plus EMA JEPA
loss. Keep source-supported laboratory channels from the original snapshot fit
patients, to avoid changing feature eligibility along with temporal encoding.
Event normalization uses only native history rows of fitting queries.

Compare native_real and native_support. Both receive identical native laboratory
histories and event update masks. Support removes normalized event numbers but
retains known masks. Include hash-bound original own/whole predictions and
persistence. Report all and bracket-after on source evaluation and held-out
hospitals with patient-level paired bootstrap; do not select on either role.

Histories retain all groups in the existing 48h window, with inert padding.
Current query inputs and labels are identical to the snapshot experiment.
Expanded projection initially preserves the lab-only native forward. Activating
rate-only updates then intentionally changes the recurrence; zero-init equality
to old snapshot predictions is not claimed. Original clipped168h clock remains
unchanged. This tests representation and event cadence jointly, not cadence
alone or verified drug administration time. Rate timestamps are record times.

Use --native-events on scripts.train_rate_anchor_jepa.py. Save checkpoints and
progress privately with native-specific names; no original checkpoint overwrite.
Inherited pretraining patient overlap remains uncertified. Seven overlapping
splits are not independent replications; no new external lockbox is evaluated.
This is the base JEPA filter diagnostic, not validation of persistent organ
architecture. No runtime source, calibrator, causal claim or promotion changes.

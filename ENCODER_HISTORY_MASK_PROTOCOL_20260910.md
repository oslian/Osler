# Empty History Step Repair

Code location: `encode_context_components` calls `NeuralBeliefFilter` without
its existing observation_mask. Availability inputs contain genuinely empty
lag snapshots. Those steps currently update the recurrent hidden state as
though a new observation arrived. A synthetic leading-empty counterexample
changes hidden state; passing the existing mask eliminates that dependence.

This is not the repeated-event identity repair: observed snapshots can still
contain the same last measurement. This trial only skips entirely unobserved
history steps, using canonical measurement mask columns. It does not deduplicate
real repeated measurements, infer event identity, change input times, or claim
measurement-process information is necessarily useless.

Use candidate belief_input_mode observed_steps: any observed variable in a
step enables the existing recurrent update; otherwise skip. Physical zero
with mask1 remains active. Empty prefix must be invariant to its length.
This laboratory-only trial has no treatment-only update steps. All scored
anchors have observed glucose, so trailing query-only decay is not tested.

Repeat original source-supported32-epoch own/whole bounded/identity experiment,
seven seeds, same source fit/tune selection including epoch0, same sampling,
optimizer, losses, raw inputs and5-7h glucose labels. No rescaling or dropout.
Evaluate full fixed validation roles, compare same original arms and own/whole
plus persistence using paired patient bootstrap. No external settings search.
Both own and whole now change, so require unchanged source input/data hashes
and seeds, not identical old predictions. Save belief_input_mode in each
candidate checkpoint because state_dict alone cannot express this behavior.

Measure empty step prevalence before fitting. No runtime promotion, interval
reuse, causal claim or pristine lockbox claim. A repaired software invariant
does not establish that it explains the MAE gap; report negative results too.

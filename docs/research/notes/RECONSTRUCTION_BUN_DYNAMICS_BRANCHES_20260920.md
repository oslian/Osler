# BUN Body Versus Organ Dynamics

Fixed bounded-site epoch7 and original90selected/3086broad source-selection
pairs. Full forward is unchanged; temporary PyTorch hooks zero body_dynamics
or organ_dynamics outputs only. All observations, update gates, absolute event
times, integration substeps, bookkeeping clock and uncertainty growth remain.
Hooks always removed, including exceptions. No weight updates.

| Group | Full | Body dynamics off | Organ dynamics off | Both off |
|---|---:|---:|---:|---:|
| Selected90 |1.105465|1.136537|1.105782|1.136537|
| Broad3086 |.893612|.851153|.880282|.851153|

Normalized masked current-reconstruction MAE, not future forecasts. Broad body
off change-.042459CI[-.048571,-.036126], about4.751%improvement. Organ off
change-.013330CI[-.015743,-.010989], about1.492%improvement. Selected body-off
and organ-off changes+.031071/+.000317 both have intervals crossing zero.
Intervals are descriptive, unadjusted, post-checkpoint-selection.

Full errors reproduce previous normal forward exactly for every patient. First
run stopped on an overly tight1e-7comparison between two no-dynamics algorithms.
The prior algorithm assigned the elapsed clock directly; hooks retain actual
integration clock arithmetic. Rerun records maximum differences4.47e-8selected,
1.19209e-7broad, below the explicit1e-5numerical check; full-model reproduction
kept its original1e-7check and is exactlyzero. No statistical gate relaxed.

## Structural Interpretation

Body-off and both-off are exactly equal per patient. This is expected from this
current-reconstruction architecture, not evidence that organ states are useless:
readout uses body state; organ state reaches it through body dynamics, while
observation body proposals feed body updates independently. Removing body
dynamics blocks that particular organ-to-readout route too.

Thus the broad BUN penalty localizes to the learned body-transition route,
partly influenced by evolved organs. It is not established as all-body-target
failure, physiological causal coupling, a software arithmetic bug or a universal
deletion solution. Selected patients do not improve. Disabling a trained path
also changes the state distribution.

Before changing training, repeat this fixed branch comparison across all33
trained targets on the original source groups. Six pending targets remain
explicit. This determines whether body-transition harm is systemic or target-
specific and whether a proposed correction would damage already useful state
dynamics. Future-trajectory requirements cannot be inferred from reconstruction.

Two focused tests pass, checking hook cleanup and expected body-off structural
isolation. No runtime, intervals, registry, weights or promotion changed.
Evidence: artifacts/native_reconstruction_bun_dynamics_branches_20260920.json.

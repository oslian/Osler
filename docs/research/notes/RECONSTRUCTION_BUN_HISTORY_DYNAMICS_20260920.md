# BUN Observation Updates Versus State Dynamics

## Controlled Replay

Same bounded-site epoch7,90selected/3086broad source-selection BUN pairs. Encode
each target/sibling-masked event once. Four replays retain the original event
grid, initialization time, update-gate delta and final anchor:

1. Full: all measurement updates and learned between-event dynamics.
2. No dynamics: all updates; hold body/organ state between events while advancing
   the bookkeeping clock. Keep original actual delta supplied to update gates.
3. Latest updates: normal dynamics on original grid, but only apply updates at
   the latest event of each independent variable. Whole simultaneous groups kept.
4. Latest plus no dynamics: combine2and3.

Unlike suffix truncation, skipping older updates does not shorten the clock or
merge transition segments. Removing dynamics also freezes uncertainty growth;
uncertainty does not feed the reconstruction readout or update gate. This is a
diagnostic intervention, not a deployable physiological prediction policy.

## Results

Every full per-patient error reproduces the prior normal forward within1e-7.
Normalized current-reconstruction MAE:

| Group | Full | No dynamics | Latest updates, fixed clock | Latest, no dynamics |
|---|---:|---:|---:|---:|
| Selected90 |1.105465|1.136537|1.149052|1.146313|
| Broad3086 |.893612|.851153|.853815|.849728|

No-dynamics-minus-full broad-.042459, descriptive95% paired interval
[-.048571,-.036126], about4.751%lower error. Selected+.031071,
CI[-.016966,+.080081], no improvement established.

After dynamics is removed, deleting older updates adds only-.001424broad
CI[-.003489,+.000650] and+.009776selectedCI[-.004492,+.025087]. Both intervals
cross zero; do not claim equivalence or no possible history benefit.

## Localization

For this fixed checkpoint and broad source population, retaining all measured
content while disabling between-event state evolution removes most of the
previously observed history penalty. This implicates the learned dynamics path,
not merely repeated observations or lost latest-variable coverage. It is
stronger evidence than inspecting gradient reachability or final update alone.

Still not a complete root cause or repair: selected population does not improve,
weights were trained with dynamics, changes affect body and organ dynamics
together, and state-distribution shift remains. Body drift versus organ-driven
drift and their interaction remain unisolated. These are source-selected,
repeatedly examined cohorts and unadjusted post-selection intervals.

Do not delete the entire dynamics path or activate a retrospectively selected
patient gate. Next localize body versus organ evolution with identical clock
and observations; any training correction must then preserve current-value
accuracy across all targets and independently verify future trajectories.

One focused replay test covers both post-anchor gaps and target-only dropped
events, checking exact full forward and single-observation factor equivalence.
No runtime weights, intervals, registry or promotion changed. All33trained and
6pending targets remain in scope; BUN is a diagnostic, not scope reduction.

Evidence: artifacts/native_reconstruction_bun_history_dynamics_20260920.json.
Private patient errors remain in ignored data/.

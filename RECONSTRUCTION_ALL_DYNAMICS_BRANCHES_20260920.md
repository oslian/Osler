# All Trained Targets: Dynamics Branch Diagnostic

Same bounded-site epoch7; original90selected and90broad source-selection groups,
2470/2025accepted patient-target pairs across33trained targets. Six pending
targets remain untrained, not implicitly passed. This is not a replay of all
3323broad source patients. All full per-patient errors reproduce1e-7against
the prior cached history audit. No fitting, route selection or new lockbox.

Apply the tested temporary hooks to zero body dynamics or organ dynamics;
retain all observations, update gates, clock, integration and uncertainty.

| Group | Full macro MAE | Body off | Organ off |
|---|---:|---:|---:|
| Selected |1.797984|1.811577|1.798333|
| Broad |1.787841|1.796712|1.786237|

Target-equal normalized masked-current-reconstruction errors. Body off harms
24/33selected and21/33broad numerically; organ off harms16/33and18/33.

Nominal unadjusted paired95% intervals, post-selection/descriptive only:

- Body off gains selected: hematocrit,lactate; harms alkaline_phos,bilirubin,
  calcium,creatinine,o2sat,ptt,total_protein,wbc.
- Body off gains broad: bicarbonate,bun,chloride,sodium; harms albumin,ast,
  calcium,o2sat,wbc.
- Organ off gains selected: fibrinogen,lactate,map,total_protein; harms
  creatinine,hematocrit,o2sat,platelets,ptt.
- Organ off gains broad: ast,bicarbonate,bun,calcium,troponin_i; harms hemoglobin.

No target has a nominal body-off gain interval excluding zero in BOTH groups.
These lists are not a source-selected runtime recipe; multiple comparisons,
small per-target support and previously selected checkpoint preclude promotion.

## Decision

The BUN diagnosis does not generalize into a global branch deletion. Shared
body dynamics can help other measurements, including calcium,o2sat,wbc in
both groups. Broad BUN harm coexists with useful dynamics, not universal failed
integration or a wholly disconnected architecture. Organ-off near-neutral
macro averages conceal target tradeoffs too.

Reject the universal no-dynamics repair. Remaining work concerns learning
state evolution that retains useful directions without corrupting others.
This diagnostic does not tell whether the harmful component is a shared offset,
state-dependent drift or supervision weakness. Inspecting the per-target
direction of the fixed branch contribution on fitting versus source-selection
patients is distinct from another global on/off or head sweep. Do not tune
branch strengths from these evaluated outcomes.

Four focused branch/summarization tests pass. No runtime, weights, registry,
interval or promotion changed. Not evidence about future horizons or causal
organ physiology. Evidence: artifacts/native_reconstruction_dynamics_branches_20260920.json.

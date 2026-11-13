# E29: Immediate Organ Assimilation Into the Same Persistent Body

## Question and falsification

E28 found that the last organ update cannot reach the same-time body-only
readout, although the same measurements already enter the body encoder.
E25 direct organ concatenation and E27 binding adaptation failed their complete
screens. This intervention is NOT another readout concatenation: it changes
observation assimilation during the entire recurrent history, keeping one body.
The established separation of continuous evolution and observation-time update
is illustrated by [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes).
We reuse the project's PyTorch state layer, not that implementation or a new ODE
library. This is a local test of timing/access, not a claimed physiological law.

Hypothesis H2-local: an immediately usable organ correction adds useful masked
current-value information beyond the existing body observation bypass.
Against it: the bypass may already suffice, the correction may duplicate noise,
or the same-time pathway may simply increase fitting without generalization.

## Fixed intervention and controls

Use the five existing E27 context sets, original epoch10 initialization, frozen
event JEPA, and the SAME binding proposals in all arms. Retain target/sibling
masks, actual known times, roles, truth, scaler and body proposals unchanged.
After the native update, form the flattened organ innovation (updated minus
pre-update organs). Add `max_state_delta * tanh(W @ innovation)` to updated body.
W is a bias-free linear projection to body width, initialized exactly zero.
No extra target head, input value skip, fake time advance or parallel body model.
This bound is in latent coordinates, not a biological conservation guarantee.

Three equally parameterized trained arms:

- immediate: current organ innovation, written at this observation time.
- delayed: preceding observed event's innovation, written at the next actually
  observed event. Pending innovation is per-replay local; never flushed at query,
  absent observation, or into another patient. The native positive-time organ
  dynamics remain active. This tests a one-observed-event delay, not equal elapsed
  time or a physiologically optimal delay. Recurrent divergence is part of the
  intervention; a difference cannot be attributed only to the very last event.
  With N observed events it applies N-1 nonempty corrections, versus N immediate
  corrections. Single-event histories do not train the delayed projection.
  Therefore this is an assimilation-policy comparison, not a pure timing effect.
- body: native body innovation, expanded by a fixed norm-preserving repeated-axis
  map to the same input width. W has identical parameter count, but effective
  input rank differs. This controls extra transformation/capacity, not identical
  information or conditioning.

The zero start must exactly reproduce unmodified binding replay. W must receive
nonzero gradients on the first step; organ-input gradients can be zero at zero
initialization and must become available after a nonzero update. Empty updates
must remain empty, and no clock change may be introduced by the correction.

## Training, roles, and fixed evaluation

Each arm starts at the ORIGINAL epoch10 checkpoint, not the best E27 arm weight.
Train layer, original body readout and W together: exactly two passes, AdamW
lr1e-4, weight_decay .01, gradient norm1, seed20261004, eval mode, deterministic CPU.
Same context order/budget as E27, no hyperparameter or checkpoint selection.
2048 fit and1024 selection contexts/fold; reused development, not a fresh lockbox.
Do not pool people across fit/selection within a fold; average repeated fold
observations per person before bootstrap. Inventory all53 variables including
unsupported cases. This bounded mechanism experiment is not full-cohort fitting.

Reuse verified E27 binding-without-bridge predictions as the unbridged matched
training control; preserve original frozen native and strong full-fit raw HGB.
Raw has more fitting support, so this is a required strong benchmark, not a pure
architecture comparison. Check all row identities/truth/hash before reuse.

Report fit and selection MAE, patient-equal and source-component-equal summaries.
Report projection saturation (absolute tanh >=.99) and correction size without
using these diagnostics to change bounds or select patients. Describe zero-gap
and single-event support separately; no post-hoc subgroup promotion.
Five fixed immediate contrasts: delayed, body, unbridged, frozen, raw.
5000 paired-bootstrap replicates, minimum30patients/3components, both95% CI upper
<0 and both one-sided p_less<.05/(53*5). No post-result retuning or exclusions.
Incremental timing evidence requires delayed AND body AND unbridged comparisons;
full research screen requires allfive. Neither is a runtime promotion.
If fit improves but selection reverses, record that position; if both new controls
improve equally, don't attribute it to organ timing. If raw still wins, complete
reconstruction repair is not achieved. A failed fixed experiment closes this
specific bridge/budget, not all observation-update mechanisms.

## Verification and boundaries

Structural tests: zero-start parity; zero/missing distinction; immediate vs
delayed last-event dependence; gradients; unchanged clocks; per-patient pending
reset; bounded correction; identical parameter counts; saved replay; bad time
rejection. All original E27/E28 tests must continue to pass.

Before accepting findings, independently replay saved predictions for both
roles, verify untouched frozen event encoder, unchanged E27 baselines, hash
bindings, paired statistics and all53 inventory. A partial run is incomplete,
never an efficacy result. Stop on hash, role, parity, nonfinite or <10GiB space.
Keep private data/candidate weights ignored and aggregate-only public reports.
No serving, release, interval, forecast, causal or complete-root-cause claim.

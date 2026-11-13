# Measurement-anchor encoder findings

## Decision

This diagnostic does not promote a runtime source. Exact target-measurement
anchors make the `creatinine@6h` task easier, but they do not rescue a stable
whole-body advantage over target-only history or persistence.

The strongest supported diagnosis is that the remaining failure is not merely
stale anchor timing. Conditional on the patient's own creatinine history, the
incremental non-target latent correction is small and changes ordering across
databases.

## Frozen comparison

The protocol was frozen before outcome inspection in
`MEASUREMENT_ANCHOR_ENCODER_PROTOCOL_20260909.md`.

- Target: `creatinine@6h`, using only observed future measurements 5-7 hours
  after the anchor.
- Fresh mode: anchor is an actual creatinine measurement; creatinine age is
  exactly zero.
- Carried mode: the existing regular 3-hour grid can carry an older
  creatinine value.
- eICU roles contain 218 fit, 93 patient-test and 78 hospital-test patients.
- MIMIC-IV uses the same 807 patients in fresh and carried modes.
- Every learned arm uses the same target-history features and fixed
  HistGradientBoostingRegressor configuration.
- The target-only and whole-body arms each have 263 features. The only
  difference is whether non-creatinine patient values enter the frozen JEPA.
- MIMIC-IV was evaluated only after the fitted eICU heads were serialized and
  hashed. It is repeated validation, not an untouched lockbox.

## Point results

Lower MAE is better.

| Evaluation | Persistence | Own history | Target-only JEPA | Whole-body JEPA | Wrong-patient latent |
|---|---:|---:|---:|---:|---:|
| eICU fresh, patient | 0.2175 | 0.2145 | **0.2117** | 0.2183 | 0.2255 |
| eICU fresh, hospital | 0.2190 | 0.2178 | **0.2106** | 0.2155 | 0.2233 |
| eICU carried, patient | 0.2950 | 0.2837 | **0.2794** | 0.2904 | 0.2965 |
| eICU carried, hospital | 0.2563 | 0.2552 | 0.2517 | **0.2457** | 0.2534 |
| MIMIC fresh | **0.1789** | 0.1839 | 0.1880 | 0.1854 | 0.1863 |
| MIMIC carried | **0.1983** | 0.2095 | 0.2054 | 0.2019 | 0.2042 |

Fresh anchoring lowers MAE for every method, including persistence. For
example, eICU patient-test persistence falls from 0.2950 to 0.2175. The fresh
and carried modes use the same patients but not the same anchor times or
outcomes, so this difference is descriptive evidence that measurement anchors
select an easier, fresher prediction task. It is not a causal estimate of an
encoder repair.

## Paired evidence

### Target-only encoder

- eICU fresh patient test is better than persistence in all seven resamples,
  but the patient bootstrap remains inconclusive: delta MAE -0.00584, 95% CI
  [-0.01716, 0.00537].
- eICU fresh hospital test is better in six of seven resamples, with delta
  -0.00838, 95% CI [-0.01922, 0.00219].
- eICU carried patient test passes the paired patient comparison against
  persistence, but not the complete care-unit gate.
- In MIMIC fresh mode it is significantly worse than persistence: delta
  +0.00910, 95% CI [0.00601, 0.01229].

This is weak source-database evidence for own-target representation, not a
transferable prediction source.

### Whole-body encoder

- In eICU fresh patient test, whole-body is worse than target-only in all seven
  resamples: delta +0.00663, 95% CI [-0.00344, 0.01673].
- In eICU fresh hospital test the ordering is mixed: delta +0.00489, 95% CI
  [-0.00508, 0.01595].
- In MIMIC fresh mode whole-body is directionally better than target-only in
  seven of seven resamples, but the paired CI still crosses zero: delta
  -0.00264, 95% CI [-0.00583, 0.00070]. It also fails to beat the
  wrong-patient placebo and is significantly worse than persistence: delta
  +0.00646, 95% CI [0.00434, 0.00863].
- No whole-body arm passes all required comparisons or supported care units.

The non-target latent therefore changes predictions, but its incremental value
is not stable across source, hospital and external database.

## Failure location

The experiment separates three possible failure points.

1. **Anchor freshness is not the complete cause.** Fresh anchors reduce raw
   error, yet persistence receives the same benefit and remains best on MIMIC.
2. **The frozen target-only representation has source-specific signal.** It can
   improve eICU point estimates, but the gain is not statistically complete and
   reverses externally.
3. **The first direct whole-body failure occurs when non-target patient state is
   added.** It does not consistently improve over target-only state and its
   ordering reverses between eICU and MIMIC.

Within this finite test, the supported problem is conditional domain
instability: after own-target history is known, the frozen JEPA/readout maps
other-organ state to small corrections whose relationship to future
creatinine is not invariant across care environments. This does not prove that
cross-organ physiology is absent. It rejects the simpler claim that stale
anchors alone hid a ready-to-use whole-body signal.

## Consequence

- Keep the current runtime, registry, calibrators and persistence fallback
  unchanged.
- Do not promote `creatinine@6h` from this diagnostic.
- Do not use another readout sweep on the same frozen latent as the next fix.
  That family has now failed under target-only, whole-body, placebo, exact-time
  and repeated external comparisons.
- A future encoder experiment must train observation semantics and
  cross-database invariance end to end from exact measurement-event histories,
  with a prespecified eICU benefit and a new untouched lockbox. It must still
  retain the target-only ablation.

The aggregate evidence is in
`artifacts/measurement_anchor_encoder_20260909.json`. Row-level examples and
fitted diagnostic heads remain outside Git.

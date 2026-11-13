# External prequential patient-state validation

> Post hoc correction: the statement below that the candidate failed all seven
> wrong-patient placebos is not supported by this artifact because its primary
> and seven-seed placebos used different donor supports. The consistent causal
> placebo rerun is recorded in `docs/research/findings/PATIENT_STATE_DIAGNOSTIC_FINDINGS_20260909.md`.
> The candidate-only decision is unchanged.

## Decision

`candidate_only`. A patient-local state was replayed prospectively for every
eligible MIMIC-IV patient. Patient-conditioned uncertainty became materially
narrower than the frozen release interval, but the time-aware point correction
did not transfer. Runtime weights, source registry, interval calibrators and
serving behavior remain unchanged.

## What changed from the previous audit

The primary slice now requires at least one prior forecast outcome to have
resolved before the current anchor. A second row without an available outcome
does not count as personalized. The shared JEPA and target head stay frozen;
each patient starts with an empty state and receives updates only from their own
resolved past errors.

The state configuration was selected on eICU patient-test rows. eICU retention
rows calibrated normalized conformal scores. eICU hospital-heldout and MIMIC-IV
were evaluated after the configuration was frozen. MIMIC-IV is repeated
external validation, not an untouched lockbox.

The selected state used a 48-hour error half-life, prior strength 1, slope
penalty 4 and at most eight resolved errors. It was compared with the frozen
release, persistence, the previous simple patient-error mean and seven
wrong-patient history placebos.

## Support

| Role | Rows | Patients | Personalized rows | Personalized patients |
|---|---:|---:|---:|---:|
| eICU patient selection | 900 | 396 | 461 | 219 |
| eICU interval calibration | 1,022 | 469 | 515 | 250 |
| eICU hospital heldout | 797 | 372 | 380 | 189 |
| MIMIC-IV replay | 2,349 | 996 | 1,231 | 436 |

## Point forecast

Patient-equal MAE on rows with at least one resolved personal error:

| Method | eICU selection | eICU hospital | MIMIC-IV |
|---|---:|---:|---:|
| Persistence | 46.06 | 40.98 | 30.51 |
| Frozen release | 45.64 | 37.06 | **29.44** |
| Simple patient state | 42.97 | **36.45** | 29.91 |
| Time-aware patient state | **42.94** | 37.76 | 31.40 |
| Wrong-patient placebo, seed 7 | 47.35 | 39.42 | 31.62 |

The eICU selection difference between time-aware and simple state was only
-0.04 mg/dL with CI `[-1.05, 0.90]`. On eICU hospital-heldout, time-aware state
was significantly worse than simple state by 1.30 mg/dL, CI `[0.32, 2.22]`,
although it still beat persistence by 3.22 mg/dL, CI `[-5.98, -0.34]`.

On MIMIC-IV, time-aware state was significantly worse than the frozen release
by 1.96 mg/dL, CI `[0.49, 3.55]`, and worse than simple state by 1.50 mg/dL, CI
`[0.73, 2.43]`. It did not significantly beat persistence. It also failed to
beat all seven wrong-patient placebos.

## Stable and changing states

The future change threshold was the pre-existing research threshold of 20
mg/dL. It was used only for retrospective evaluation, never for routing.

In MIMIC-IV stable rows, persistence MAE was 8.52 mg/dL, frozen release 16.50,
simple state 15.77 and time-aware state 15.48. In changing rows, frozen release
was best at 45.25 mg/dL; simple state rose to 47.30 and time-aware state to
50.37, while persistence was 56.04.

The time-aware state reduced false moves on stable MIMIC rows from 20.1% for
the frozen release to 14.0%. It simultaneously reduced direction accuracy on
changing rows from 68.9% to 60.4%. The patient state therefore behaved mainly
as a stabilizer. It did not learn a transferable acute glucose trajectory.

## Calibrated uncertainty

Method-specific normalized conformal radii were fit on eICU retention patients
only.

| Method | eICU hospital width | Coverage row / patient | MIMIC width | Coverage row / patient |
|---|---:|---:|---:|---:|
| Frozen release | 182.86 | 0.884 / 0.907 | 182.86 | 0.950 / 0.953 |
| Simple patient state | **160.75** | 0.876 / 0.900 | 159.39 | 0.942 / 0.946 |
| Time-aware patient state | 167.25 | 0.882 / 0.895 | **157.10** | 0.933 / 0.934 |

The time-aware interval was significantly narrower than the frozen release by
15.60 mg/dL on eICU hospital-heldout and 25.75 mg/dL on MIMIC-IV. It was wider
than the simple state interval on eICU and not significantly narrower on
MIMIC-IV. MIMIC coverage missed the declared upper bound of 0.93 by
0.003-0.004, so the narrower interval is not eligible for release.

## Failure location

The useful patient signal is mostly a local error level and a state-dependent
uncertainty scale. The slope and temporal extrapolation selected in one eICU
patient role did not remain valid in another hospital role or in MIMIC-IV.
eICU selection contained more changing than stable personalized rows, whereas
MIMIC-IV contained more stable than changing rows. The selected correction
therefore favored a regime that did not transfer.

This rules out the claim that a richer time-aware state automatically learns a
patient's changing physiology from a few measurements. The next state model
must first predict, from anchor-time information alone, whether a personal
error correction remains valid in the current regime. Until that gate passes,
the frozen release remains the point source and patient-conditioned intervals
remain research-only.

Authoritative aggregate report:
`artifacts/patient_state_external_glucose_6h_20260909.json`.

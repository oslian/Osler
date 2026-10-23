# Two-Level All-Target JEPA Findings (2026-08-01)

## Question

Can one connected model use a first-layer whole-body latent, a second layer of
organ-specific latents, and the already successful target-specific sparse
expert method for every available physiological target?

## Architecture

The experiment uses one model, not independent per-target models:

~~~text
measurement-pure multiscale patient history
    -> shared whole-body context latent
    -> 11 disjoint canonical organ latent tokens
    -> 53 target/horizon sparsemax expert gates
       (shared slow, target fast, own organ)
    -> target-specific bounded residual heads
    -> target-specific distributional scale heads
    -> patient-cluster conformal interval
    -> persistence/validated-router fallback
~~~

The point predictor is trained first. Every point-prediction parameter is then
frozen before the uncertainty scale heads are trained. The no-expert ablation
uses the same trained adapter and capacity.

## Data and support

- eICU canonical event examples;
- 1,000 stays, 922 patients, 12 hospitals;
- 127,404 rows;
- 53 targets;
- horizons: 1, 3, 6, 12, 24, and 48 hours;
- seven patient-held-out seeds.

Forty-seven targets had enough measured future labels for at least one
target-horizon cell. Six remained support-limited in this cohort:
amylase, crp_hs, free_t4, myoglobin, serum_osmolality, and uric_acid.
They were recorded as support-limited rather than silently removed or treated
as successful.

## Bounded patient-heldout result

There were 192 evaluable target-horizon cells. Eleven passed the teacher,
persistence, and matched no-expert point gates in all seven splits:

- alkaline phosphatase at 1h and 3h;
- anion gap at 1h;
- glucose at 1h, 3h, and 6h;
- heart rate at 3h, 6h, and 12h;
- magnesium at 1h and 3h.

Only three also passed patient-equal 90% distributional conformal coverage in
all seven splits:

| Cell | Median delta vs teacher | Median delta vs no-expert |
|---|---:|---:|
| glucose@1h | -0.2785 | -0.0990 |
| glucose@3h | -0.3322 | -0.1441 |
| glucose@6h | -0.4082 | -0.1840 |

## External gate

glucose@1h and glucose@3h passed:

- hospital-heldout;
- care-unit-heldout;
- forward-time heldout;
- teacher and persistence patient bootstrap;
- module-balanced accuracy;
- distributional conformal coverage;
- matched no-expert bootstrap;
- direction accuracy, delta correlation, and latent SVD stability.

glucose@6h passed the hospital and forward-time checks but failed care-unit
teacher/persistence accuracy and conformal coverage. It remains fallback.

The final research registry therefore validates:

~~~text
glucose@1h
glucose@3h
~~~

## Interpretation

The two-level hierarchy is viable. Applying the method jointly to all targets
added a validated glucose@1h cell while retaining glucose@3h. The result also
rejects a stronger claim: the same architecture does not make every target
predictable. Point-only near wins frequently failed uncertainty, and six
sparse targets lacked enough future supervision.

The model remains target-gated. Failed cells continue to use their existing
validated router, persistence, or missing. Cross-organ causal attribution,
treatment effects, clinical use, and whole-checkpoint promotion remain
disallowed.

## Artifacts

- teacher_anchored_two_level_all_targets_smoke.json
- teacher_anchored_two_level_all_targets_1000_7seed.json
- teacher_anchored_two_level_all_targets_external_1000.json
- teacher_anchored_two_level_all_targets_validated_registry_20260801.json

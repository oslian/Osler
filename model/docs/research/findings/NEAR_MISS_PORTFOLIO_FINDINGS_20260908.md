# Near-Miss Portfolio And State-Routing Findings

Date: 2026-09-08

## Decision

Combining the closest heterogeneous candidates does **not** add a promotable
source under the current tabular pre-anchor information contract.

- Fixed patient-bootstrap convex portfolio: `0/4` cells pass.
- State-dependent expected-error router: `0/4` cells pass.
- Outcome-aware oracle: `4/4` cells show a large upper-bound improvement.
- Active checkpoint, source selector, intervals and runtime remain unchanged.

This is a useful distinction. The candidate set contains a better answer after
the future outcome is known, but the available patient state cannot identify
that answer prospectively. The oracle is therefore evidence of retrospective
choice opportunity, not deployable physiological information.

## Fixed convex portfolio

Weights were fit on separate tune patients and frozen before patient-heldout and
hospital-heldout evaluation. Exact duplicate release/persistence sources were
removed without reading outcomes.

| Cell | Frozen portfolio | Patient MAE: portfolio / best single | Hospital MAE: portfolio / best single | Result |
|---|---|---:|---:|---|
| MAP@3h | 78.2% current-cross, 18.3% cross-history, 3.4% own-history | 10.256 / 10.271 | 8.872 / 8.870 | No significant gain; stable cases harmed |
| Creatinine@6h | 74.0% own-history, 26.0% cross-history | 0.2581 / 0.2590 | 0.2578 / 0.2585 | Differences vs best and placebo cross zero |
| Bicarbonate@12h | 73.7% current-cross, 26.3% own-history | 2.139 / 2.145 | 2.400 / 2.378 | Hospital reversal; stable cases harmed |
| BUN@24h | 100% cross-history | 4.997 / 4.997 | 4.737 / 4.737 | Collapsed to one source; no combination gain |

All four portfolios beat persistence on point MAE in both evaluation roles.
That does not answer the experiment's question: none significantly beats the
tune-selected best source and its capacity-matched control in both roles while
also satisfying the stable-case guard.

The candidates are less diverse than their names imply. Pairwise signed-error
correlations range from `0.907` to `1.000`; median correlations are `0.946` to
`0.997` depending on cell and role. Averaging highly correlated errors cannot
create a large robust gain.

## Outcome-oracle upper bound

The impossible row-level oracle chooses the source with the smallest error only
after seeing the true future value. Relative to the tune-selected best single
source, patient / hospital patient-equal MAE changes were:

| Cell | Patient delta | Hospital delta |
|---|---:|---:|
| MAP@3h | -2.472 | -2.350 |
| Creatinine@6h | -0.0429 | -0.0434 |
| Bicarbonate@12h | -0.530 | -0.496 |
| BUN@24h | -1.051 | -0.815 |

Every paired patient-bootstrap interval excluded zero. This does not authorize
the oracle. With several correlated predictions, outcome-aware minimum-error
selection can exploit small row-level noise differences that are unknowable at
prediction time.

## State-dependent router

The second experiment used only target-own history, candidate signed proposals,
their absolute sizes and their disagreement. One equal-capacity expected-error
model per source selected persistence or a candidate separately for each
patient. A matched placebo router replaced only cross-patient information.

| Cell | Oracle agreement patient / hospital | State-router MAE patient / hospital | Fixed-portfolio MAE patient / hospital |
|---|---:|---:|---:|
| MAP@3h | 20.2% / 18.5% | 10.218 / 8.918 | 10.256 / 8.872 |
| Creatinine@6h | 25.5% / 25.0% | 0.2589 / 0.2584 | 0.2581 / 0.2578 |
| Bicarbonate@12h | 26.7% / 20.9% | 2.127 / 2.431 | 2.139 / 2.400 |
| BUN@24h | 23.6% / 25.9% | 5.098 / 4.770 | 4.997 / 4.737 |

Agreement is approximately chance for four or five sources. MAP and bicarbonate
improve slightly in patient-heldout data but reverse in held-out hospitals.
Creatinine and BUN do not improve on the fixed portfolio. No cell beats all
references, and the real router does not consistently beat its placebo.

## What failed first

The failure is not the arithmetic operation used to combine predictions.

1. Candidate residuals are extremely correlated, so a fixed blend has little
   independent error to cancel.
2. The future outcome reveals substantial source complementarity, but legal
   pre-anchor state does not predict which source will be correct.
3. State-dependent routing therefore follows unstable row-level noise and
   changes behavior across hospitals instead of approaching the oracle.

Blindly adding more near-miss candidates would increase retrospective oracle
performance while also increasing selection overfit. It is not a valid path to
promotion under the same input contract.

## Remaining justified path

A future portfolio retry requires a genuinely new prospective discriminator of
state, not another combination rule over the same predictions. Examples are
pre-event high-frequency signals, semantically richer bedside assessment, or
treatment-intent execution state that is known before the forecast anchor. The
new signal must first beat target-own history and a matched wrong-patient
placebo in raw source attribution. Only then should it be used to route the
existing candidates.

## Artifacts

- `artifacts/near_miss_portfolio_20260908/report.json`
- `artifacts/near_miss_state_router_20260908/report.json`
- `whole_body_jepa/near_miss_portfolio.py`
- `scripts/audit_near_miss_portfolio.py`
- `scripts/audit_near_miss_state_router.py`
- `tests/test_near_miss_portfolio.py`

Both reports are aggregate-only, bind the active release and input hashes, and
set `promotion_allowed: false`, `runtime_modified: false`, and
`causal_claim_allowed: false`.

The protocol's secondary MIMIC-IV replay was not executed: the locally
available 6h transition artifact lacks the strict actual-measurement-time label
metadata used by this primary experiment, and the other predeclared horizons
are not locally available under that contract. Mixing nominal and strict labels
would have made the comparison uninterpretable. No external-transfer conclusion
is drawn from this run.

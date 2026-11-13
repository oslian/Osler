# Continual self-learning findings

Protocol frozen and evaluated on 2026-09-09. The aggregate report is
`artifacts/continual_self_learning_20260909_v2.json`.

## Question

Can the frozen shared whole-body JEPA support a small online residual adapter
that learns from newly resolved measurements without forgetting prior patients?

This is a prequential diagnostic, not a promoted runtime policy. Predictions
were recorded before the corresponding future creatinine measurement was
allowed to update a model.

## Design

- Target: `creatinine@6h`, restricted to actual future measurements 5-7 hours
  after an exact creatinine measurement anchor.
- Representation: frozen 256-dimensional JEPA latent plus seven measured
  creatinine-history features.
- Base predictor: one fixed source-trained HistGradientBoosting delta head.
- Continual component: a small bounded linear residual adapter.
- Development: eICU patients split into base-head, initial-adapter, retention,
  patient-stream and hospital-heldout roles.
- Repeated external evidence: 910 MIMIC-IV patients. MIMIC-IV is already
  inspected and is not an untouched lockbox.
- Seven deterministic patient arrival orders were evaluated.
- The three weight-updating arms received the same 16-example update budget per
  resolved label.

The compared variants were frozen global prediction, frozen prediction plus a
patient running residual state, naive online updating, short-plus-long replay,
uniform-history replay and persistence.

## Support

| Role | Patients | Rows |
|---|---:|---:|
| eICU base-head fit | 151 | 270 |
| eICU initial adapter fit | 50 | 90 |
| eICU retention | 50 | 104 |
| eICU online patient stream | 120 | 206 |
| eICU hospital-heldout | 92 | 151 |
| MIMIC-IV repeated validation | 910 | 2,021 |

Only 37 stream patients contributed second-or-later predictions, so the
patient-personalization slice remains modest.

## Results

Mean MAE across seven patient orders on second-or-later predictions:

| Method | MAE | Mean MAE delta vs frozen | Significant vs frozen | Significant vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 0.1578 | - | - | - |
| Frozen global | 0.1994 | 0 | - | 0/7 |
| Frozen patient state | 0.1954 | -0.0084 | 0/7 | 0/7 |
| Naive online | 0.1951 | -0.0206 | 0/7 | 0/7 |
| Dual replay | 0.1832 | -0.0291 | 4/7 | 0/7 |
| Uniform replay | 0.1865 | -0.0275 | 5/7 | 0/7 |

All three online arms had a lower mean error than the frozen model in every
patient order. Replay was stronger and more stable than naive updating. It did
not, however, beat persistence in any order under the paired patient-level
bootstrap criterion.

Naive online updating produced detectable post-stream harm in 3/7 eICU
hospital tests and 7/7 MIMIC tests. Dual replay reduced this to 0/7 eICU
retention, 0/7 eICU hospital and 3/7 MIMIC harms. Uniform replay produced 0/7,
0/7 and 2/7 harms respectively. Replay therefore reduces catastrophic
forgetting, but does not establish database-invariant continual adaptation.

## Decision

There is a source-side continual-adaptation signal: resolved measurements can
improve a small adapter, and replay is materially safer than naive online
updating. The full test fails because:

1. persistence remains more accurate for this short-horizon creatinine task;
2. improvement over the frozen model is not significant in all seven orders;
3. post-stream MIMIC performance remains order-dependent and sometimes
   significantly worse.

The result is `candidate_only`. Runtime weights, registry, calibrators and
serving behavior were not changed. Full JEPA online gradients are not justified
by this screen.

## Interpretation

Self-learning is not the missing universal cure for the current model. It can
adapt a residual head and replay can retain prior knowledge, but it cannot
create prospectively identifiable six-hour creatinine dynamics that the
pre-anchor state does not contain. The next valid continual-learning test would
need a target where the frozen source already competes with persistence, more
patients with repeated resolved outcomes, and a newly untouched external
database. It must retain the same prequential timing and replay controls.

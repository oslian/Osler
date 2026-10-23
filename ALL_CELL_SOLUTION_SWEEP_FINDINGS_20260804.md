# All-cell solution sweep findings (2026-08-04)

## Question

Can the accumulated whole-body JEPA solutions be applied systematically to
all supported target-horizon cells, rather than only to a few hand-picked
examples, and do any additional cells pass the full factual runtime gate?

## Contracts and methods tested

The sweep kept the existing fail-closed policy. Test labels were never used
for training or calibration, every candidate remained anchored to the
validated teacher/persistence source, and every result remained factual-only.

The following accumulated solutions were tested:

- two-level whole-body and organ latents (`hierarchical`);
- target-horizon sparse experts (`sparse_distributional`);
- target-specific future projection (`sparse_target_distributional`);
- horizon-balanced target supervision (`sparse_horizon_balanced_distributional`);
- patient-cluster cross-fitted adaptive conformal intervals;
- administration-grade, strictly pre-anchor treatment context;
- interval-normalized urine-output semantics;
- distributional shape-adaptive conformal calibration;
- irregular-event forecasting for sparse laboratory targets.

## Coverage

### Standard as-of contract

- 193 evaluable target-horizon cells across 33 targets.
- Four model variants, for 772 evaluated method-cell combinations.
- Three-seed discovery was followed by seven-seed confirmation for every new
  discovery candidate and then hospital, care-unit, and temporal gates.

### Aligned-treatment contract

- 189 evaluable target-horizon cells across 35 targets.
- Three model variants, for 567 evaluated method-cell combinations.
- Strictly pre-anchor medication/infusion context was included; no future
  treatment was available to the model.

### Shape-adaptive calibration

- Re-evaluated all 189 aligned-treatment cells with trained distributional
  scale and shape-adaptive conformal calibration.

### Sparse irregular-event contract

- 21 target-time-bin cells across seven sparse laboratory targets.
- Five value models passed patient/care-unit/forward-trajectory gates.
- Two INR cells (`6-24h`, `24-72h`) also passed conformal calibration.
- They remain outside the joint runtime catalog because an independent
  external-database gate for the exact sparse-event contract is absent.

## New validated result

Only one additional system cell survived every required gate:

- `o2sat@12h`
- source: target-specific sparse JEPA with aligned treatment context;
- patient gate: 7/7 vs teacher, persistence, and no-expert ablation;
- conformal gate: 7/7;
- hospital, care-unit, temporal, module-balanced, direction, delta-correlation,
  bootstrap, and SVD gates: passed;
- runtime input contract:
  `joint_asof_whole_body_state_aligned_treatment_interval_urine.v1`;
- unit: `%`;
- causal and clinical claims: disabled.

The materialized evidence is in:

- `all_cell_solution_sweep_o2sat12_validated_registry_20260804.json`
- `whole_body_runtime_candidate_catalog_v5.json`

Catalog v5 contains 34 unique validated system cells, up from 33 in v4, with
no previously validated cell removed.

## Important rejections

- `platelets@1h/@3h` passed the seven-seed patient gate but failed external
  transport. Shape-adaptive calibration did not rescue `platelets@3h`.
- `urine_output@6h` passed the standard internal target-projection gate, but
  failed hospital/care-unit conformal coverage and expert ablation. Switching
  to the interval-normalized aligned-treatment contract did not rescue it.
- `o2sat@6h/@24h`, hemoglobin, hematocrit, WBC, calcium, magnesium, albumin,
  total protein, potassium, and temperature produced internal or three-seed
  near-misses, but did not satisfy all seven patient splits plus external
  gates.
- Sparse hormones, troponin, coagulation, and related labs are better posed as
  irregular-event tasks. Fixed-horizon failure was not relabeled as success.

## Conclusion

Applying all accumulated fixes broadly did produce a real new result, but it
did not turn the whole grid into a promotable monolithic model. The dominant
remaining boundary is external/domain stability, not a missing single neural
architecture trick. The correct runtime remains target-gated: use each exact
validated source only under its matching input contract and unit; otherwise
fall back to persistence or missing.


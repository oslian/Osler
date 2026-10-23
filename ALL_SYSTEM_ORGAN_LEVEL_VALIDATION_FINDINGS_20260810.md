# All System-to-Organ Level Validation (2026-08-10)

## Scope

This audit evaluates the corrected adjacent hierarchy:

```text
functional system <-> anatomical organ <-> whole body
```

It does not treat functional systems as organs and contains no direct
system/body path.

- source: eICU canonical event ledger;
- 1,000 patient stays and 124,350 target-horizon rows;
- 53 targets;
- 11 functional systems;
- 14 anatomical organ/structure nodes;
- horizons: 1h, 3h, 6h, 12h, 24h, and 48h;
- seven patient-held-out seeds under one joint 53-target training problem;
- hospital-, care-unit-, forward-time-, bootstrap-, path-ablation-, and
  patient-cluster conformal gates for patient-level candidates.

Primary artifacts:

- `adjacent_three_level_strict_anatomical_all_systems_1000_7seed_20260810.json`;
- `adjacent_three_level_strict_anatomical_all_systems_external_gate_all_candidates_20260810.json`.

## Validation meanings

Three levels must not be conflated:

1. **Structural mapping**: the legal system-to-organ edge exists in the model.
2. **Patient-level signal**: the full hierarchy and an organ/body path beat the
   validated teacher, persistence, and system-only prediction in seven of
   seven patient splits.
3. **Externally stable cell**: the same cell also passes hospital, care-unit,
   forward-time, conformal, module-balanced, and matched path gates.

All 11 systems pass level 1. Seven systems contain at least one level-2 cell.
No cell passes level 3.

## Full validation matrix

| Functional system | Anatomical organs | Targets | Evaluable cells | 7/7 point + organ/body path | Patient candidates | External pass |
|---|---|---:|---:|---:|---|---|
| Cardiovascular perfusion | heart, blood vessels, kidneys | 4 | 14 | 7 | heart_rate@12h; map@3h; map@12h | none |
| Respiratory exchange | lungs, heart | 2 | 12 | 5 | respiratory_rate@24h | none |
| Renal clearance | kidneys | 4 | 18 | 6 | none | none |
| Acid-base/electrolyte | kidneys, lungs | 11 | 54 | 12 | calcium@1h | none |
| Endocrine/metabolic | pancreas, liver, thyroid, adrenal glands | 5 | 7 | 2 | none | none |
| Hepatic synthesis/detoxification | liver | 8 | 36 | 3 | none | none |
| Digestion/absorption | stomach, intestines, pancreas, liver | 2 | 0 | 0 | none | none |
| Hematopoiesis/coagulation | bone marrow, liver, spleen | 6 | 36 | 0 | none | none |
| Immune/inflammation | bone marrow, spleen, liver, brain | 6 | 12 | 3 | none | none |
| Cardiac injury | heart | 2 | 4 | 0 | none | none |
| Musculoskeletal injury | skeletal muscle, kidneys | 3 | 0 | 0 | none | none |

There were 193 evaluable target-horizon cells. Thirty-eight cells had a 7/7
point and organ/body-path improvement. The remaining nominal cells were absent
from evaluation because their exact future-label support did not satisfy the
audit contract.

## External gate results

All five patient candidates selected the body-mediated path. Direct
organ-to-organ coupling remained disabled for these candidate paths.

| Cell | Closest verified result | Blocking result |
|---|---|---|
| heart_rate@12h | Passed patient, hospital, time, point/path, and most care-unit gates | care-unit conformal |
| calcium@1h | Passed hospital and care-unit point/path gates | hospital conformal and forward-time gates |
| respiratory_rate@24h | Passed hospital and care-unit point/path gates | hospital conformal and forward-time gates |
| map@3h | Passed patient and hospital gates | care-unit point/path and module-balance gates |
| map@12h | Passed patient and hospital gates | care-unit point/path, conformal, and module-balance gates |

`external_gate_passed_cells` is empty. No cell is eligible for runtime
promotion.

## Organ observability boundary

The anatomical graph contains heart, blood vessels, lungs, kidneys, liver,
pancreas, stomach, intestines, thyroid, adrenal glands, bone marrow, spleen,
brain, and skeletal muscle.

Stomach, intestines, and spleen have no direct first-class numerical variable
in the current canonical event schema. They can receive a functional-system
message, but the current experiment cannot independently verify their latent
state. Thyroid and adrenal-gland observations exist but are sparse. These
nodes therefore pass structural validation only.

## Conclusion

The system-to-organ layer is implemented correctly and contains reproducible
patient-level information in seven of eleven functional systems. The strongest
signals occur in cardiovascular, respiratory, renal, acid-base/electrolyte,
endocrine, hepatic, and immune targets. However, none is stable across every
patient, hospital, care-unit, time, uncertainty, and path-ablation gate.

The honest current status is:

- structural system-to-organ coverage: complete;
- bounded patient-level predictive evidence: partial;
- externally validated system-to-organ forecast cell: zero;
- promotion status: `candidate_only`;
- `causal_claim_allowed: false`.

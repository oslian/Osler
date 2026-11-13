# Hidden Physiology Belief Findings

Date: 2026-07-12  
Scope: candidate-only predict-update proxies

## Implemented states

`hidden_physiology_beliefs.py` adds two online states:

- `glycemic_clearance_tendency`: inferred from the patient's observed glucose
  trajectory and factually observed insulin context when present.
- `acid_base_buffer_reserve`: inferred from observed bicarbonate, pH, anion
  gap, PaCO2, lactate, freshness, and recent trajectory.

Each state emits a mean, uncertainty, innovation, confidence, and observation
availability. The state is updated in stay/time order and never reads future
suffix columns.

These are deliberately not called insulin sensitivity or buffer-pool truth.
The first is not a causal pharmacodynamic estimate; the second is not a direct
measurement of total buffer capacity.

## Downstream gate

The eICU audit compared:

- baseline ridge using observed state/action features;
- baseline plus both hidden-state features;
- a capacity-matched placebo with the same feature width.

It used seven patient-heldout seeds and a hospital-heldout split. The states
must improve measurable downstream targets against both baseline and placebo.

Result: **0 target cells passed the full gate** in the tested endocrine
cohort. Some individual splits showed partial signal, but the hospital-heldout
result was not stable. Both states therefore remain `candidate_only`.

## What remains validated

The existing AKI renal reserve / creatinine kinetics belief remains the only
fully validated hidden-state component in this branch. Existing validated
beliefs in cardiovascular, electrolyte, respiratory, endocrine, immune, and GI
modules remain governed by their own findings and gates; this experiment does
not overwrite them.

## Safety boundary

- No direct hormone-concentration accuracy claim.
- No insulin-sensitivity causal claim.
- No direct buffer-reserve accuracy claim.
- No counterfactual or clinical authority.
- No promotion into the active runtime until downstream observable gates pass.


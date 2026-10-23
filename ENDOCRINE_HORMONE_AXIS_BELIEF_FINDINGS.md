# Endocrine Hormone-Axis Belief Candidate

Status: candidate-only. This experiment adds inferred control-loop states to
the existing endocrine belief implementation. It does not measure or predict
a hormone concentration directly.

## States Added

### Water-balance / ADH-like drive

The online state uses current and prior trajectory information from:

- sodium and serum osmolality;
- urine output when the cohort contains it;
- creatinine and MAP;
- fluid/dextrose context when observed.

The output is an inferred water-regulation drive with uncertainty. It is not
an ADH lab result and must not be described as ADH concentration.

### Insulin-response state

The online state uses:

- glucose trajectory slope and innovation;
- an observed insulin-context flag;
- anion-gap/glucose freshness;
- observed steroid and dextrose context.

The output is an inferred response state. It is not insulin concentration or a
causal estimate of insulin sensitivity.

## Audit Design

Both states use the existing predict-update feature implementation and compare:

1. baseline Ridge using the observed transition features;
2. candidate Ridge plus the new hormone-axis features;
3. capacity-matched placebo Ridge plus random features.

The audit uses seven patient-heldout seeds and hospital-heldout evaluation. A
state can only be promoted when it beats both baseline and placebo on every
required patient split and the hospital-heldout check.

## Results

### Insulin-response candidate

The endocrine stress cohort contained 17,009 transition rows, 1,149 subjects,
1,420 stays, and 12 hospitals. The candidate showed a near-miss for MAP:
hospital-heldout performance passed, but only 6/7 patient splits passed. No
target passed the complete promotion rule. Glucose improved in some splits but
was not stable enough for promotion.

### Water-balance candidate

The electrolyte/acid-base cohort contained 17,387 transition rows, 1,249
subjects, 1,385 stays, and 26 hospitals. It included urine output, so it was
used for the water-balance audit. No downstream target passed the complete
seven-split plus hospital-heldout rule.

## Decision

Both new states remain candidate-only. The negative result does not prove that
ADH or insulin physiology is uninformative. It shows that these cohorts and
the current 6-hour observation contract do not provide enough stable evidence
to promote the inferred states.

The next valid improvement would require denser hormone measurements,
administration-grade insulin/fluid doses, longer horizons, or a cohort with
repeated endocrine assays. It would be incorrect to solve this by forcing the
candidate states into the validated whole-body forecast layer.

## Safety Boundary

- no direct hormone concentration claim;
- no causal or counterfactual treatment claim;
- no clinical or runtime authority;
- no checkpoint promotion;
- candidate features may be used only for shadow research and future audits.

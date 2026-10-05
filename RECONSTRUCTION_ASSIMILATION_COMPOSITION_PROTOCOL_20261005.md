# E32: fixed E29-to-E31 common-source decomposition

Question: does the adaptation fit-to-same-source mean-gap arise only from
source-component weights, or do error differences remain inside common sources?
E30 could not answer this; E31 supplies the missing role. No new predictions,
training, calibration, significance testing, threshold search or promotion.

## Fixed comparison

Use all 53 inventory targets and five original folds, independently. Compare
E29 fitting patients with E31 adaptation-unseen patients. Exclude none by outcome.
Use immediate minus each of unbridged (primary), delayed, body and frozen
absolute error. Each fold has one query per patient. Preserve every unsupported
target-fold, role count, component count and common-support fraction.

On common components only, renormalize each role's patient weights to one.
Let e/f denote E31/E29 and m the component mean paired error difference:

```
gap = sum(w_e*m_e) - sum(w_f*m_f)
within = sum((w_e+w_f)/2 * (m_e-m_f))
composition = sum((w_e-w_f)/2 * (m_e+m_f))
gap = within + composition
```

Reuse `audit_crossfit_group_shift.decompose`; map its selected/broad roles
explicitly to E31/E29. Independently verify with the two directional
standardizations averaged together. Also report the all-source gap separately;
its difference from common-support gap is a restriction/renormalization effect,
not a measured counterfactual for unsupported sources.

Report anonymized component ordinals, role counts/means, minimum counts and
number of common components with fewer than30 people in either role. No
posthoc dropping of small components or reporting only favorable targets.
Cross-fold counts are descriptive, NOT independent replications. Do not pool
folds to manufacture within-model overlap. No causal interpretation: within
source still confounds patient mix, observation support and estimation noise.

## Falsification and checks

- Pure weight-change synthetic examples must have zero within term.
- Pure within-source change at fixed weights must have zero composition term.
- A reversal induced by weights alone must be separated from a within shift.
- Missing role/no shared source returns unsupported, never a fabricated zero.
- Reject patient role overlap, changed provenance/target identity, duplicate
  queries and nonfinite outcomes/predictions.
- Require complete hash-bound E29/E31 verification before loading predictions.
- Check output arithmetic independently from saved anonymous component rows.
- Do not change existing model weights, gates, artifacts or runtime.

If within terms remain adverse, component reweighting alone is insufficient for
these observed means. If composition dominates, that is a descriptive lead,
not permission to fit a correction on these evaluation outcomes. Both terms
can cancel. Insufficient support remains a limitation under either outcome.

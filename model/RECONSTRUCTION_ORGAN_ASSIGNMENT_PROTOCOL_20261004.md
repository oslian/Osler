# E26: Organ observation assignment invariance

## Question and scope

Does the native organ proposal discard the binding between an observed variable
identity and its numeric value before recurrent updates? This is distinct from
the prior source-event ID collision audit and from swapping whole patient states.
No new patient outcome fitting, head search, or release change is authorized here.

For a fixed observed organ member set S, the current proposal is

`mean_i(phi(value_i, mask_i, age_i) + variable_embedding_i + module_embedding_i)`.

Permuting value/mask/age triples among observed members of S leaves this sum
unchanged. Variable identity exists in the representation of the set, but is not
bound to each value. This claim is only about the pre-mixer organ path. The body
encoder has another path and may distinguish these inputs. It does not establish
the principal cause of the complete masked-reconstruction or forecast gap.

## Fixed diagnostic

- Use all five E25 epoch10 mappings, explicitly reporting fold2 = source fold0.
- Check every pair in the 53-variable schema sharing an anatomical owner.
- Values are normalized synthetic pairs (-1,1), (0,1), (1,2); no clinical claim.
- Observe either just the two variables or every member of their organ.
- Keep identity slots, masks, other values, event time and units fixed. Age is zero
  for observed entries and seven for absent entries. Swap only the pair's values.
- Reproduce the native organ proposal from its primitive operations; evaluate
  in float64 with fixed absolute tolerance 1e-10, zero relative tolerance.
- Compare body proposals, native organ proposals, and native organ states after
  update and six hours of evolution. Native body feedback is disabled; record it.
- Structural repair candidate: apply existing PyTorch SiLU to the combined
  numeric and identity token BEFORE pooling, rather than after discarding the
  binding. No new learned parameter or external data is used. Reuse the same
  update/dynamics only as a sensitivity check, not accuracy evaluation.
- Controls: identical-value exchange, record reordering with identity attached,
  missing-value sanitation, absent-organ isolation and finite outputs.
- No candidate selection, outcome score, confidence interval or promotion.

## Interpretation and stopping

Nonzero native organ change above tolerance rejects the claimed exact invariance
for that case. A native collision with a changed body localizes a path-specific
limitation, not total information loss. Candidate separation removes this exact
symmetry on tested inputs; it is neither injectivity proof nor a forecasting fix.
Do not silently replace pretrained inference with the candidate: training and
compatible calibration are required. No hyperparameter sweep follows this audit.

## Established pattern

Deep Sets (Zaheer et al., 2017): transform each complete set element, including its
identity and value, before invariant aggregation. Reordering complete observations
should not matter; exchanging values while identities stay fixed should matter.
https://arxiv.org/abs/1703.06114

The proposed parameter-free SiLU binding is a diagnostic instantiation of this
pattern, not a claim that one activation guarantees a sufficient physiology model.

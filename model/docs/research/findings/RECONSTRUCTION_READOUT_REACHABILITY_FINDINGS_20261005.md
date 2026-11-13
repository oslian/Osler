# E42: no hard readout-rank bottleneck among supported targets

This is a saved-weight algebraic audit of E40, not training, a physiology test,
a future forecast audit or a promotion. Exploratory inspection preceded the
written protocol; no claim of a preregistered unseen hypothesis is made.

## Result

| Scope | Rows | Rank in each saved fold | Interpretation |
|---|---:|---:|---|
| Entire catalogue | 53 | 47 | Six exact-zero heads must not be hidden |
| Nonzero readouts | 47 | 47 | No hard affine dependence between these outputs |
| E41 scored targets | 46 | 46 | Same conclusion on scored scope |
| E41 sufficiently supported targets | 27 | 27 | Hard output dependence does not explain their failed screens |

Exact-zero weight and bias rows: ck_mb, crp, crp_hs, ionized_calcium,
prealbumin and urine_output. All six are unscored and unsupported in E41;
none is one of its 27 sufficiently supported failures. Ferritin is also
unscored but has a nonzero head. These are reconstruction heads, not the
release's separate forecast capabilities. Do not deploy their zero outputs
as estimates or randomly initialize them and claim missing support is fixed.

Across saved folds, nonzero-row condition numbers are40.51–54.01; on the27
supported rows they are15.45–19.49. Independent NumPy and torch ranks agree.
For the supported subset, float32 right-inverse unit-output replay error is
at most3.5763e-7. Thus numerical rank deficiency is not imposing an exact tie
between their outputs. Condition numbers alone do not prove optimization is
easy or all training scales are appropriate.

## What this rules out, and what it does not

The sole explanation "there are not enough independent linear output directions"
is rejected for these supported heads. Adding more heads merely to repair that
alleged deficiency is not supported by this audit.

This does not establish that the recurrent encoder can infer or attain a
desired body state under its bounded updates. The pseudoinverse uses prescribed
unit coordinate changes, not patient information, and is never a deployable
prediction rule. It does not test the reachable-state manifold, information
retention, conditional transfer, uncertainty or clinical validity.

The six zero heads account for the whole-matrix rank deficiency, not the27
supported predictive failures. Full generalization repair remains unresolved.

## Verification

- All five original E40 checkpoint and contract hashes match saved bindings.
- E41 is bound to the same E40 report; inventory and original support preserved.
- 15 audit bindings rechecked, including code, tests and protocol.
- Seven unittest tests pass, including collinear, zero, nonfinite and malformed
  support cases. The initial diagnostic assumed every unsupported JSON record
  had criteria; corrected to honor E41's explicit unsupported schema. This was
  an audit implementation error, not an original model root cause.
- No pytest dependency was installed; tests use existing standard unittest.
- Fold0/2 are not independent replications. No new external data or patient
  rows published. Runtime, gates, weights and intervals unchanged.

[Protocol](../../../RECONSTRUCTION_READOUT_REACHABILITY_PROTOCOL_20261005.md)
and [aggregate results](../../../artifacts/native_readout_reachability_20261005.json).

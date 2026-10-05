# Variable-Specific Observation And Residual Transport

## Question And Difference From Prior Work

Does similar aggregate support conceal different observed variables, or
opposite within-component numeric/residual associations? This follows the
low-overlap support-matching diagnosis. It does NOT repeat the September17
raw-summary ridge versus body readout trial, which did not support wholesale
replacement of the encoder. No new regression/head is fitted here.

## Exact Inputs

Extracted all1,800,214 masked patient-target histories into private variable
snapshots. Identity/order exactly matches the verified support cache. Every
payload hash is checked. All53 schema slots retain separate value, observed
mask and actual measurement age. Missing values/ages are NaN, observedzero
remains observed. Target/sibling masks are inherited from the immutable cache.

Latest measurement is selected using anchor minus event availability plus
token measurement age, not simply the last received result. Exact measurement
age ties use the latest cached available version. This is an analysis snapshot,
NOT a change to encoder ingestion. No future label or target truth determines
features. The private Parquet is not committed.

## Comparison

All235target-folds assessed. Six have an empty role (myoglobin in allfive,
pHfold4);229 yield12,137variable comparisons. Source calibration versus
component-disjoint outer data use the same frozen predictions and patients.
Already inspected outer data are reused validation, not an untouched lockbox.

Observation/freshness fractions use all patients with equal component weight.
Numeric associations use observed patients in components with>=30observations;
at least3supported components are required in EACH role. Values and outcomes
are centered within component, then covariance/variances pooled with equal
component weight. Retained observation counts are explicit. This removes
between-component means from the correlation, not confounding or case-mix
within components. It is not exact state matching or causal attribution.

## Results

- 857of12,137variable comparisons have an absolute observed-fraction difference
  of at least20percentagepoints. This is a descriptive count, not significance.
- 5,160comparisons support finite residual correlations in both roles.
- 1,291have opposite signs, but only4have absolute correlations>=0.1 on BOTH
  sides. Of531comparisons with both absolute correlations>=0.1,527retain sign.
  The0.1summary is not a gate, feature-selection rule or statistical proof.
  Many tiny correlations flip; sign counts alone would exaggerate instability.

Examples of component-equal observation fraction shifts (outer minus source):

| Reconstruction / fold | Input variable | Shift |
|---|---|---:|
| albumin /2 | anion_gap | -0.357 |
| albumin /2 | heart_rate | +0.237 |
| albumin /2 | temperature | +0.210 |
| o2sat /3 | respiratory_rate | -0.217 |
| map /4 | troponin_i | -0.147 |

Albumin's aggregate input counts/freshness were similar in the preceding
diagnosis, yet variable identities differ materially. Counts alone therefore
do not establish equivalent observation states. This observation is NOT proof
that removing or imputing these variables will improve reconstruction.

Four descriptive stronger sign flips: magnesium/bilirubin_direct fold1,
serum_osmolality/creatinine fold1, amylase/bicarbonate fold4,
respiratory_rate/bilirubin_direct fold4. No feature deletion or routing decision
is made from these outer findings. Residual correlations can reflect omitted
variables, miscalibration, selection or noise; they do not isolate the encoder's
response to an intervention on one input.

## Decision

Confirmed: aggregate support hid variable-specific observation shifts.
Not supported as a blanket explanation: all numeric physiology relationships
reverse across domains. Unresolved: which observation shifts actually cause
prediction harm, and whether source-only conditional correction transfers.

The next repair needs a source-only, component-held-out validation of a
conditional correction versus identical process-only and intercept controls,
with its inputs fixed before seeing any more outer results. A prior small-cohort
raw ridge failure must remain the counterexample; these correlations alone do
not justify another universal readout or deleting weak variables. The snapshot
now permits that diagnosis without re-extracting or altering histories.

Seven new tests cover measurement-versus-availability ordering, revisions,
zero/missing, hidden targets, truth-independent features, within-component
centering, and explicit insufficient/constant support. Two previous support
tests also pass. No model training, runtime, interval, forecast or promotion
changed. This remains masked CURRENT reconstruction, not future JEPA efficacy.

Artifacts:
- `artifacts/native_reconstruction_variable_snapshot_20260925.json`
- `artifacts/native_reconstruction_variable_transport_20260925.json`

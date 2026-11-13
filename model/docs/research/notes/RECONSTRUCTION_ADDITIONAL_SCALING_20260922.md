# Additional-Domain Fit-Only Scaling

## Result

Recovered all 1,800,214 accepted current labels for 73,670 people from the
canonical raw event shards. Every accepted label is finite and unique per
patient/variable. All five predefined folds have patient-equal median/IQR
scalers fitted exclusively on fit patients. Replacing all non-fit values with
1e12 leaves every scaler array exactly unchanged.

The 53-variable schema has six unsupported label channels in every fold:
CK-MB, CRP, hsCRP, ionized calcium, prealbumin and urine output. This is a
statement about the present qualified extraction, not absence in all raw data.
Missing centers remain NaN; unsupported channels require masking/abstention.
Zero is an observation. Zero IQR uses one canonical unit, not an inferred
physiological variance. No held-out values determine scaling.

## Defect Found Before Use

The first synthetic regression test failed: the correlated NOT EXISTS SQL
returned a later conflicting value despite a previously available inconsistent
measurement. An explicit left anti-join against known conflicting events fixes
that counterexample. Five tests pass, covering future conflict visibility,
conflict-before-latest selection, invalid latest draws, revision rejection and
held-out mutation invariance. The deeper DuckDB binding/optimizer cause is not
established. This newly written extraction was not used in historical training;
the defect cannot explain previous model failures.

The fast path explicitly rejects multiple versions of a source event. The full
run passed this restriction, exact accepted-label extraction, shard hashes and
partition dependency checks. Single-version extraction is not a replacement
for general source-version replay.

## Boundary / Next Work

Follow-up parity verification: twelve randomized mixed lab/bedside event
streams with delayed, invalid and conflicting observations reproduce the
reference reconstruction builder's accepted requested-target values exactly
(more than 100 comparisons). Six scaling tests plus three reconstruction
support tests pass together. This extends synthetic label-value parity, not
full-dataset re-encoding. All scaling artifact bindings and upstream masked
history code/event-index bindings also match the current files.

These are raw-unit current-value reconstruction labels, not future labels or
accuracy results. The old experiment used normalized history summaries in some
cases; raw MAE numbers must not be compared across those scales. No encoder,
readout or interval calibrator has been trained by this step. Fit-only input
vocabulary support and exact masked training histories remain to be prepared
before matched physiology/process training. No runtime weights, registry,
intervals or promotions changed.

Report: `artifacts/native_reconstruction_additional_domain_scaling_20260922.json`.
Patient-level centers and scaler tensors remain in ignored private data.

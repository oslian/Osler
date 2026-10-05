# Raw Probe Scaling Warning Diagnosis And Repair

Replayed all33 targets x body/raw/process probes using exact hash-bound current
body/raw caches. Original scores reproduce within1e-8. Recorded weighted
StandardScaler warnings and variance attributes, then replaced only exact
source-constant column handling. Nonconstant columns still use StandardScaler;
PCA, Ridge100, weights, masks, labels and roles are unchanged.

32 sqrt warnings correspond to288 negative variance entries; every such entry
belongs to an exactly constant source column. Minimum variance is about
-9.70e-35. Original final scales are all finite because sklearn's constant
feature handling replaces the invalid intermediate scale after the square root.
This is floating-point cancellation, not variable input values being NaN.

Repair: detect exact constancy from fitting rows only, emit zero centered
training values and query-minus-source-constant with unit scale. Fit the
standard library scaler on remaining columns. RuntimeWarnings on those
remaining columns raise an error rather than being suppressed. Both all-
constant and mixed-column regression tests pass.

Full repaired33-target replay completes without scaler RuntimeWarnings.
Maximum individual prediction difference is1.44e-11; maximum reported MAE
difference is2.03e-13. Thus this numerical issue did not materially explain the
previous raw-versus-body result. It is not a predictive improvement or proof
that all preprocessing is correct. Prior report/source files stay immutable;
the tested helper is available in scripts/audit_raw_probe_scaling.py.

No runtime model, registry, intervals or promotion changed. Two tests and
diff check pass. Evidence:
artifacts/native_reconstruction_raw_probe_scaling_20260920.json.

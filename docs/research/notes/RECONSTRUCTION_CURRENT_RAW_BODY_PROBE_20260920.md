# Current Dimension-Matched Raw Versus Body Probe

Executed fixed RECONSTRUCTION_CURRENT_RAW_BODY_PROTOCOL_20260920.md. All14,150
source patients, all33trained targets,247611fit/77219selection pairs. Raw
histories joined to current bounded7 body cache by exact patient/target/truth,
with input hashes and complete feature digest. Same existing target/sibling
mask. Private per-patient raw caches support later diagnostics without replay.

Raw265/process265 become64 via fit-weighted standardized PCA, then fit-score
standardization. Body64 receives fit standardization. Same Ridge100 solver and
source weights. No new hyperparameter or target selection. This is dimension-
matched linear readout capacity, not equal representation learning procedures.

## Results

| Normalized macro MAE | Original | Body ridge64 | Raw PCA ridge64 | Process PCA ridge64 |
|---|---:|---:|---:|---:|
| Selected |1.797984|3.197392|3.237073|3.108499|
| Broad |1.559771|2.708317|2.668227|2.713592|

Raw beats body numerically on14/33targets in each group. No target passes the
existing incremental patient/site gate against body, process and original.
The plain ridge probes are overall much worse than original; changing from
the trained robust objective to squared-error ridge is not a proposed runtime
replacement. Large sparse-target errors affect these macro averages.

| Target/group | Original | Body probe | Raw probe |
|---|---:|---:|---:|
| Glucose/selected |1.04239|1.16256|1.20742|
| Glucose/broad |.83329|.93528|.99219|
| PTT/selected |2.09004|1.90858|2.44914|
| PTT/broad |1.37559|1.41562|1.67105|
| BUN/selected |1.10547|1.06754|1.05815|
| BUN/broad |.89361|.86156|.84885|

## Interpretation And Limits

No stable raw-summary advantage establishes encoder information loss as the
general cause. BUN has small numerical raw advantages, but no full gate gain.
Glucose/PTT favor body over raw in bothgroups. This does not prove body is
sufficient: unsupervised PCA may discard predictive low-variance directions;
latest/mean/lag/count summaries do not retain detailed sequence ordering;
fixed ridge100 may underfit or react poorly to heavy-tailed targets. Fitting
another hyperparameter from these outcomes would not make a fresh validation.

Runtime emitted sklearn weighted StandardScaler sqrt-variance warnings.
All33 fits completed and every reported MAE is finite; this alone does not
establish the exact origin of each warning. Constant-column numerical handling
should be audited before using these transforms as production artifacts. No
such transforms or probe weights were deployed. Do not silently equate a
completed diagnostic with a validated inference contract.

Two tests pass: query changes cannot alter source PCA/scaling, and incorrect
body dimension is rejected. Audit exit0. No release weights, intervals, registry,
forecast ability, causal claim or promotion changed. Reused source selection,
not end-to-end encoder crossfit or new external lockbox.

Evidence: artifacts/native_reconstruction_current_raw_body_probe_20260920.json.

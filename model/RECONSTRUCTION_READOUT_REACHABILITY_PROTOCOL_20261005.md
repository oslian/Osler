# E42: saved readout algebraic reachability

Retrospective diagnostic, not a new model or efficacy test. Exploratory matrix
inspection already found six zero rows and rank47 in each E40 saved readout;
this protocol records that inspection, not a prospectively unseen hypothesis.

Question: does a hard affine dependency between output rows explain the failed
supported targets? Compare all53 rows, the exact nonzero rows, scored rows,
and E41 support-qualified rows. Keep all five saved folds and all zero heads.
Do not infer that full row rank makes the encoder informative or its states
reachable under actual recurrent dynamics.

Use float64 NumPy SVD with tolerance max(shape)*eps*largest singular value.
Report rank, singular values, condition number (null for rank deficiency),
right-inverse output projector error in float64 and float32, and minimum-norm
state changes for unit output-coordinate changes. Those unit directions are
algebraic probes, not patient observations or a physiological intervention.
Cross-check with independent torch matrix_rank/pinv. Bind E40 contracts and
weights against their completed audit; take target support from fixed E41.

Failure condition: supported output rows are rank deficient, or constructing
their right inverse fails numerical replay. If they are full row rank and the
unit probes reproduce numerically, reject hard output dependency as the sole
explanation for those failures. Conditioning can remain an optimization issue;
this audit cannot declare it solved or require arbitrary whitening.

This differs from the 20260920 spectrum audit: that audit measured covariance
of observed body features and removed weak feature directions from a newly fit
readout. This audit examines the frozen output map itself. No rank search,
fitting, changed gates, patient exclusions or runtime edits. No promotion.
MIMIC/NWICU remain reused development and folds0/2 are not independent.

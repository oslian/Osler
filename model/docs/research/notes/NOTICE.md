# Data and model notice

This repository contains a factual forecasting research artifact derived from
credentialed MIMIC-IV and eICU data. It contains no row-level patient records.

The trained weights and fitted target heads must not be made public or shared
with anyone who is not authorized under the applicable PhysioNet data use
agreements. A public release of derived models should be distributed through
PhysioNet under the same agreement as the source data. Keep this GitHub
repository private.

This artifact is not a medical device, is not clinically validated, and must
not be used to make treatment decisions. It supports factual forecasting only;
all outputs set `causal_claim_allowed` to `false`.

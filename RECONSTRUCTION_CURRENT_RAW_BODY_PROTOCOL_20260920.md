# Current Raw Versus Body Probe Protocol

All33 trained targets, all current source-fit/encoder-selection patients, exact
body features from bounded7 and exactly matching raw history shards. Reuse
the existing target-and-sibling masking and raw latest/mean/presence/lag/count
summary. No future labels/target history. Do not reuse old encoder body features.

Raw265 and process265 standardized using source fit only, weighted source PCA
to64, then source-only score standardization. Body64 standardized source-only.
Same Ridge alpha100/SVD readout, same expected bounded-site/equal-group weights
scaled to mean1. Fixed settings, no source-selection tuning. Original head is
an additional reference, not capacity-matched trained baseline.

Compare raw against body, process and original with existing patient/site gate.
No candidate installation or fresh external claim. Raw PCA can discard useful
low-variance directions; summaries omit sequence order. Negative results do
not establish absence of raw information. Different representation learning
procedures remain despite matched linear head dimension. This is current-value
masked reconstruction, not prediction of future values or causal physiology.

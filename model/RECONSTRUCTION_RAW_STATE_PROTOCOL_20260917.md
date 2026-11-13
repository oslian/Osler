# Matched Raw History Versus Frozen Shared State

Use the existing strict patient-linked site fit/tune/evaluation split and all33
supported targets, retaining six unsupported entries. This differs from the
small earlier frozen-latent probe: full strict-site cohort, identical masked
native events and a raw-history comparator. Reused evaluation is development
evidence; warm-start independence remains uncertified.

After identical target/all-sibling masking, summarize each variable's latest
value, mean value, observed flag, latest availability lag(hours), log1p event
count. Missing numerical values zero with flag0 and lag49 (48h lookback).
The process comparator zeros both numerical blocks but retains support/time.
Another comparator uses the SAME selected real model's body immediately before
readout; verify original evaluation predictions. No encoder update.

Fit StandardScaler+Ridge on fit only, alphas1/10/100/1000/10000; choose each
target/source on tune MAE against fit-median baseline, which wins ties. Readout
training/selection procedure matches but input dimensions differ; this is not
capacity-matched proof of encoder information loss. No test-conditioned routing.

Report all target/source MAEs and nominal patient bootstrap2000(seed20260917)
against population, original shared readout and raw-process. No promotion,
causal, future forecast or conformal claim. A favorable raw-vs-state contrast
motivates representation repair, not a claim raw data guarantees generalization.

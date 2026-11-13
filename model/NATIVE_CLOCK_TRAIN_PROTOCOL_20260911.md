# Matched clock repair

Use --native-events --repair-clock on the existing runner. Same source-supported
channels, query context, labels, patients, seven partitions, seeds,32epochs,
16steps/batch32, tune-only selection and loss as native baseline. Only history
clock columns change from clipping[0,168] to signed unbounded hours, scaled
by24/168 as before. This changes both learned observation and decay clock, not
only decay. Within48h histories the existing decay-gap cap of168h is not active.

Re-infer original native checkpoints on original inputs and assert published
MAEs before comparison. Real/support variants retain the same event masks and
source-only event scaler. Primary paired contrast is repaired versus original
native, then support, original snapshot whole, own and persistence. Evaluate
all/bracket-after source evaluation and hospital with existing patient bootstrap.
No selection on these results; no new lockbox, intervals or promotion. Private
clock-prefixed weights/progress preserve all prior outputs. Warm-start overlap
and retrospective treatment time limitations remain. Runtime unchanged.

# Fixed Extended Training Budget

Question: does the eight-epoch budget explain the remaining whole-vs-own gap?
Repeat all four source-supported bounded/identity x own/whole arms with seven
seeds,32 epochs instead of8. Identical optimizer, loss, source split, sampling,
input hashes and source-tune checkpoint selection including epoch0. First eight
epoch logs must exactly reproduce the previous confirmation; mismatch means
the contrast is not isolated. No external selection of epoch or arm.

Compare paired patient errors against the SAME arm from the eight-epoch run,
and whole vs own plus persistence, on the same scored patients. Save all local
prediction arrays, report all contrasts and seven-seed95% bootstrap CIs.
Longer training allows more source-tune checkpoint comparisons; apparent tune
improvement alone is not evidence of generalization. Both domains are already
inspected validation. No new calibrator, runtime change, promotion or untouched
lockbox claim. Use the existing descriptive subgroup protocol for follow-up.

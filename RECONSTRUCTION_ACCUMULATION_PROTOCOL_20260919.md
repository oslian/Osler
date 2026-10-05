# Balanced Patient Accumulation

Keep the same expanded-source33head architecture, initialization, eight epochs,
draw queues, source priors and90+90selection patients. Each epoch has536rounds,
eachround one observation per33targets x2groups. Accumulate8rounds=528examples,
average loss gradients, clip once then AdamW once. Same283008example exposures
as sequential reference, but536optimizersteps instead of283008. Do not call
this equal optimizer compute or a pure variance-only intervention.

Two predeclared rates:1e-4 and1e-4*sqrt(528). Source-only macroMAE selects epoch
including0and chooses one rate before evaluation. Square-root scaling is a
sensitivity arm, not an equivalence theorem for AdamW. Weight decay per step,
moment estimates, clipping and step count differ. Do not select rate on evaluated
cohorts. Preserve sequential reference and its existing controls; new candidate
requires matching process/placebo validation before any promotion claim.

Eight distinct patients pergroup-target occur within each block unless a queue
crosses its cycle boundary; larger sample than single-example but not the64patient
gradient probe. Full64patient/task training and six untrained heads remainpending.
All39statuses retained. No runtime, registry, calibration or causal changes.

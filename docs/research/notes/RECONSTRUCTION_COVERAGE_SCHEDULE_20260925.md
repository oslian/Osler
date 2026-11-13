# Coverage-Aware Schedule Engineering Check

Implemented weighted no-replacement cycles per target using NumPy choice.
Cycle remainder plus RNG state are resumable; complete each pool before repeat.
Same47targets,1072draws/target/epoch,eight epochs and50384updates/epoch.
No labels used to choose ordering. Three new tests and five existing sampling
tests pass, including uninterrupted versus resumed stream equality.

Replay of ORIGINAL replacement sampler exactly reproduces checkpoint `seen`
sets for allfourindependentfits, not merely matching counts.

|Fold|Original unique pairs|Coverage schedule unique pairs|Eligible|
|---|---:|---:|---:|
|0|236509|296652|698766|
|1|236632|296623|698510|
|3|230194|290051|682000|
|4|243475|305209|743818|

Approximately25-26%more unique pairs at identical update count. This still
does not cover the entire cohort, because equal target budgets exhaust rare
targets and revisit them while common targets have much larger pools.

Important limitation: weights now control ordering; once a full cycle is
completed, frequency is uniform. This is not identical marginal sampling to
replacement. Any downstream improvement needs matched training, not attribution
solely to reduced duplicates. Same old baseline cannot be called independently
replicated just because its schedule was reproduced.

This is a tested sampler and exposure simulation ONLY. No encoder trained,
accuracy improvement, outer result, runtime change or promotion yet.
Artifact: artifacts/native_reconstruction_coverage_schedule_20260925.json.

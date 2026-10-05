# Raw-Event Multi-Target Reconstruction Builder

Implemented native_reconstruction_examples.reconstruction_examples. Reuses
the conflict-checked as-of scheduler on original measurement events, at each
requested anchor. No union of cached later query histories. Fresh labels must
arrive exactly at that anchor and pass existing encode_group quality/age rules.
Every example hides target and sibling measurements throughout its history;
no independent remaining history means no example. Encounters remain separate.
Availability/unit qualification remains the source extractor's responsibility.

Tests:14 pass across native reconstruction examples, body masking and history
scheduler. New counterexamples check future revisions do not rewrite earlier
truth, known conflicting labels rejected, cross-encounter observations excluded,
and empty new-event anchors do not manufacture carried labels.

Real-source smoke completed on first8 sorted source-fit patients:1,877 raw
availability anchors produce6,249 masked current-target examples. Examples
include MAP966, heart_rate1039, glucose500, creatinine96, bicarbonate130,
potassium114. Counts are examples, not independent patients or validated cells.
Several sparse targets have zero examples in this small sample. No population
coverage claim follows. The builder took several minutes; it is not yet a
full-cohort training cache or a trained multi-target neural model.

This repairs the new-example construction route; prior single-query native
forecasts were not shown invalid. No accuracy/promotion/runtime change.
Weights, calibrators and registry unchanged. Source-only smoke, not heldout.

Evidence: artifacts/native_reconstruction_builder_20260913.json, with bindings
to the builder, scheduler, masking and original event inputs. No patient rows
or identifiers saved in the aggregate artifact.

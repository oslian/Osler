# Full-source Expansion Inventory

Original eICU patient table contains139,367patients/200,859stays. Using existing
hospital roles without moving any current patient, and excluding current cohort,
current ancestral-source patients and sites, the following additional patients
have ALL original-table stays contained within a single existing role:

| Role | Additional patients | Stays |
|---|---:|---:|
| Fit | 10201 | 12926 |
| Encoder selection | 3323 | 3989 |
| Readout confirmation | 3433 | 4131 |
| Evaluation | 7178 | 9866 |
| Reserved | 16372 | 22140 |

These are provenance-only upper bounds, NOT qualified reconstruction examples.
No lab availability, analyte support, quality, availability delay or mask contract
has been checked. No patients were assigned, extracted into model inputs, trained,
or scored. Reserved/evaluation patients must remain excluded from fitting.
Historical use elsewhere in this project has not been excluded; not new lockbox.

## Additional provenance limitation

49 current patients have full original-table hospital histories extending outside
their assigned role, including potentially unassigned hospitals. The existing
partition verified query/extracted-history sites, not every raw-table admission.
This does NOT establish those outside stays entered a model or crossed actual
training/evaluation inputs. Patient roles remain unique. Do not label this as
proven leakage, blanket-invalid release, or the cause of accuracy failures.
Before cohort expansion, freeze the boundary explicitly: full-table patient/site
components versus strictly selected input histories. Do not silently add those
outside stays under the current partition.

## Implication

The 89-patient confirmation set is not the demonstrated limit of locally available
patients. A larger label-qualified confirmation cohort is feasible to investigate
without reassigning current evaluated patients. Next check source-only lab/event
eligibility across all39targets and exact isolation before another training run.
Do not promise that additional patients will improve performance.

Evidence: `artifacts/native_reconstruction_expansion_inventory_20260917.json`.
No runtime, interval or promotion changes.

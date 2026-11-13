# Potassium order coverage and alias repair

Fixed6174 source queries, no outcomes/model fitting/external data. Broad lexical
screen of original infusion and medication names followed by existing classifier.
V1 aggregate preserves before-repair evidence; V2 binds repaired classifiers.
This is not exhaustive medication concept recall: the lexical screen itself
can miss unknown names. No patient rows or identifiers exported.

| Source | Cohort rows | Eligible screened rows | Classified before/after | Prior6h query windows before/after |
|---|---:|---:|---:|---:|
| infusionDrug | 78568 | 65 | 65 / 65 | 20 / 20 |
| medication | 82538 | 4133 | 3993 / 4133 | 727 / 739 |

Medication screen found4387rows;254marked canceled are excluded from the
descriptive eligible count. Missing availability/cancellation history prevents
calling this a prospectively known set. Cancellation flags may be finalized
after the query: do not reuse this filtered table as safe historical features.

The140missed order names were KLOR-CON65, K-Dur2036, K-DUR35,
and KLOR-CON,K-TAB4 ("K-Dur 20" is a name, not inferred administration dose).
Added the two evidenced aliases to existing potassium terms; no dose/route
inference or runtime input change. The infusion classifier missed none of the
screened infusion names, so aliases do NOT explain the preceding infusion
experiment's low support or negative prediction result.

Verified identity sources:
- [DailyMed Klor-Con potassium chloride](https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=1ff53330-065c-4213-9c0c-ac498621d09d)
- [FDA K-Dur reference drug identity](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2014/206814Orig1s000OtherR.pdf)

The larger gap is source scope: orders appear in739query windows versus20
infusion windows. These may overlap; they are not additive and do not prove
739actual administrations. Previous lab-plus-infusion trial is not a full
medication-intent test. An order-only channel would need its own availability
and cancellation semantics before an accuracy trial. Never relabel dosage
or planned start as delivered dose.

Artifacts: `artifacts/potassium_medication_coverage_20260911.json` (original)
and `artifacts/potassium_medication_coverage_20260911_v2.json` (repaired).
Runner: `scripts.audit_potassium_medication_coverage --output <fresh-path>`.
No promotion, model retraining, interval or serving changes. Extraction recall
repair is established; forecast benefit remains untested.

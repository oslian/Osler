# MAP Pressor Capture Inventory and Unit Repair

The current expanded native MAP candidate contains laboratory/bedside inputs
and insulin event context, not a complete treatment history. Outcome-free rate
inventory on its33,477eligible queries/1,505patients found substantial omitted
pressor documentation. This is input-scope evidence, not forecast efficacy.

## Reproduced and Fixed Parser Defect

`canonical_rate` supported units/hour but omitted units/minute. Explicit
`Vasopressin (units/min)` therefore became unknown. Added singular/plural
min/minute aliases with exact factor60 into drug_units_per_hour. Two tests
failed before the fix; all13 focused tests pass afterward. Documented zero is
preserved by rate_channels; ml/hour is solution flow, never guessed drug dose.

Within the same cohort, vasopressin compatible rate rows82->13,922;
queries with a documented compatible rate in preceding6h10->2,101;
patients2->231. All-drug union9,776->9,954queries and768->775patients because
many recovered vasopressin records coexist with other pressors. Other four
drug inventories are unchanged.

After repair norepinephrine support is8,231queries/667patients;
epinephrine864/154, phenylephrine1,812/214, dopamine734/62.
Drug identities and weight-normalized dimensions are not summed or converted
using guessed weights/concentrations.

## Limits and Next Boundary

The infusion table has infusionoffset but no independent availability timestamp.
Pre-anchor recorded timing is not verified live knowledge or active/delivered
dose. The original MAP native trial did not consume these pressors, so this
parser defect is NOT an established cause of its own/full accuracy gap.
Connecting the repaired observations requires an explicit retrospective timing
contract or independently verified availability, new training, and matched
validation. No accuracy gain or promotion has been established.

The reusable parser changed; existing trained caches, weights, release registry
and calibrators were not replaced. New extractions can differ. Historical
artifact hashes remain historical, not retroactively current-code verification.
Before report omits the transitive treatment_units hash; repaired report binds it.

Evidence: `artifacts/map_pressor_inventory_20260912.json` (before),
`artifacts/map_pressor_inventory_repaired_20260912.json` (after).

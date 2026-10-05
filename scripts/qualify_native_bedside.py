"""Keep rejected observations as explicit quality evidence, not deleted history."""
import json
from pathlib import Path
import pandas as pd
from extractors.eicu_sepsis_transition_extract import PLAUSIBLE
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    source = root/'canonical_bedside_events.parquet'
    destination = root/'qualified_bedside_events.parquet'
    output = Path('artifacts/native_bedside_quality_20260911.json')
    if destination.exists() or output.exists():
        raise ValueError('fresh outputs required')
    events = pd.read_parquet(source)
    events['quality_valid'] = False
    summary = {}
    for variable, part in events.groupby('variable'):
        lower, upper = PLAUSIBLE[variable]
        valid = part.value.between(lower, upper)
        events.loc[part.index, 'quality_valid'] = valid
        summary[variable] = dict(rows=len(part), accepted=int(valid.sum()),
            rejected=int((~valid).sum()), lower=lower, upper=upper)
    events.to_parquet(destination, index=False)
    report = dict(complete=True, training_performed=False, runtime_modified=False,
        policy='existing extractor bounds; invalid finite observations retained with quality_valid=false',
        variables=summary, bindings={str(p):sha256(p) for p in [source,destination,Path(__file__),
            Path('extractors/eicu_sepsis_transition_extract.py')]})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

"""Audit whether E29 can separate unseen patients from unseen sources."""
import json
from pathlib import Path

import pandas as pd

from scripts.diagnose_organ_assimilation import require_verification
from scripts.train_organ_assimilation import DEST, validate_identities
from scripts.train_organ_binding import write_json
from whole_body_jepa.multisource_intervals import sha256


def overlap_counts(fit, selection):
    common = set(fit.component) & set(selection.component)
    return dict(fit_occurrences=len(fit), selection_occurrences=len(selection),
        fit_components=int(fit.component.nunique()),
        selection_components=int(selection.component.nunique()),
        common_components=len(common),
        fit_common_occurrences=int(fit.component.isin(common).sum()),
        selection_common_occurrences=int(selection.component.isin(common).sum()),
        comparison_status=('missing_role_support' if fit.empty or selection.empty
                           else 'same_source_support_exists' if common
                           else 'same_source_comparison_unavailable'))


def audit_roles(fit, selection, inventory):
    if not inventory or len(set(inventory)) != len(inventory):
        raise ValueError('Unique nonempty inventory required')
    for frame in (fit, selection):
        validate_identities(frame)
        if frame.empty or not set(frame.variable) <= set(inventory):
            raise ValueError('Empty roles or unexpected variables')
    # Repeated patients across folds are legal; provenance must remain fixed.
    joined = pd.concat([fit, selection], ignore_index=True)
    if (joined.groupby('patient').component.nunique() > 1).any():
        raise ValueError('Patient provenance changed across roles/folds')
    if (joined.groupby('variable').context.nunique() > 1).any():
        raise ValueError('Target context changed across roles')
    if (joined.groupby('context').variable.nunique() > 1).any():
        raise ValueError('Context target changed across roles')
    folds = sorted(set(fit.fold) | set(selection.fold))
    result = {}
    for fold in folds:
        a, b = fit[fit.fold == fold], selection[selection.fold == fold]
        if a.empty or b.empty:
            raise ValueError('Both roles required in every fold')
        if set(a.patient) & set(b.patient):
            raise ValueError('Within-fold patient role overlap')
        result[str(fold)] = dict(**overlap_counts(a, b), patient_overlap=0,
            targets={v: overlap_counts(a[a.variable == v], b[b.variable == v])
                     for v in inventory})
    statuses = [v['comparison_status'] for f in result.values() for v in f['targets'].values()]
    return dict(folds=result, summary=dict(
        folds=len(folds), inventory_targets=len(inventory),
        folds_with_common_components=sum(f['common_components'] > 0 for f in result.values()),
        target_fold_status_counts={s: statuses.count(s) for s in (
            'missing_role_support', 'same_source_support_exists',
            'same_source_comparison_unavailable')},
        pooled_component_overlap_not_within_model=len(set(fit.component) & set(selection.component)),
        runtime_modified=False, promotion_allowed=False))


def main():
    ap = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    vp = Path('artifacts/native_organ_assimilation_verification_20261004.json')
    protocol = Path('RECONSTRUCTION_ASSIMILATION_ROLES_PROTOCOL_20261005.md')
    out = Path('artifacts/native_organ_assimilation_roles_20261005.json')
    if out.exists():
        raise ValueError('Completed audit exists; do not overwrite')
    assessment, verification = json.loads(ap.read_text()), json.loads(vp.read_text())
    require_verification(ap, verification)
    paths = (ap, vp, protocol, Path(__file__),
             Path('scripts/diagnose_organ_assimilation.py'),
             Path('scripts/train_organ_assimilation.py'),
             Path('scripts/train_organ_binding.py'),
             Path('tests/test_assimilation_roles.py'))
    bindings = {str(p): sha256(p) for p in paths}
    frames = {r: [] for r in ('fit', 'selection')}
    for fold in range(5):
        rp = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
        if sha256(rp) != assessment['bindings'][str(rp)]:
            raise ValueError('Changed fold report')
        report = json.loads(rp.read_text())
        bindings[str(rp)] = sha256(rp)
        for role in frames:
            path = DEST/f'fold{fold}'/f'{role}_predictions.parquet'
            digest = sha256(path)
            if digest != report['prediction_hashes'][role]:
                raise ValueError('Changed prediction file')
            frame = pd.read_parquet(path)
            if set(frame.fold) != {fold}:
                raise ValueError('Prediction fold mismatch')
            frames[role].append(frame)
            bindings[str(path)] = digest
    result = audit_roles(*(pd.concat(frames[r], ignore_index=True) for r in ('fit', 'selection')),
                         inventory=list(assessment['results']))
    result['bindings'] = bindings
    result['limitation'] = ('Role identifiability only. No overlap cannot distinguish in-sample '
        'optimism from source/patient/observation shift. Pooled fold overlap is not same-model '
        'overlap. No third-role inference, training, new gate or causal attribution.')
    write_json(out, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()

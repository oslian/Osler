"""FHIR R4 structural validation for the resources this demo ingests.

This is not a full profile validator. It checks the elements the pipeline
depends on (ids, meta versioning, status value set, codings, subject, effective
time, value/dataAbsentReason exclusivity per obs-6, obs-7) and parses times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

OBSERVATION_STATUS = frozenset({
    "registered", "preliminary", "final", "amended", "corrected", "cancelled",
    "entered-in-error", "unknown",
})
BUNDLE_TYPES = frozenset({"searchset", "collection", "transaction", "batch", "history"})
CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/observation-category"


class FhirValidationError(ValueError):
    def __init__(self, issues: list[str]):
        super().__init__("; ".join(issues))
        self.issues = issues


@dataclass(frozen=True)
class EffectiveTime:
    effective: datetime           # clinical time (period end for a Period)
    period_start: datetime | None
    period_hours: float | None


def parse_instant(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise FhirValidationError([f"{field} must be an RFC3339 string"])
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise FhirValidationError([f"{field} is not RFC3339: {value}"]) from exc
    if parsed.tzinfo is None:
        raise FhirValidationError([f"{field} must carry a timezone offset: {value}"])
    return parsed.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return iso(now_utc())


def _require(resource: dict[str, Any], field: str, issues: list[str]) -> Any:
    value = resource.get(field)
    if not isinstance(value, str) or not value:
        issues.append(f"{resource.get('resourceType', '?')}.{field} is required")
        return None
    return value


def _check_meta(resource: dict[str, Any], issues: list[str]) -> None:
    meta = resource.get("meta")
    if not isinstance(meta, dict):
        issues.append(f"{resource.get('resourceType')}.meta is required (versionId, lastUpdated)")
        return
    if not isinstance(meta.get("versionId"), str) or not meta["versionId"]:
        issues.append("meta.versionId is required")
    try:
        parse_instant(meta.get("lastUpdated"), "meta.lastUpdated")
    except FhirValidationError as exc:
        issues.extend(exc.issues)


def _check_reference(resource: dict[str, Any], field: str, expected_type: str, issues: list[str],
                     required: bool) -> None:
    ref = resource.get(field)
    if ref is None:
        if required:
            issues.append(f"{resource.get('resourceType')}.{field} is required")
        return
    reference = ref.get("reference") if isinstance(ref, dict) else None
    if not isinstance(reference, str) or not reference.startswith(expected_type + "/"):
        issues.append(f"{field}.reference must be a relative {expected_type}/<id> reference")


def reference_id(resource: dict[str, Any], field: str) -> str | None:
    ref = resource.get(field)
    reference = ref.get("reference") if isinstance(ref, dict) else None
    return reference.split("/", 1)[1] if isinstance(reference, str) and "/" in reference else None


def validate_patient(resource: dict[str, Any]) -> None:
    issues: list[str] = []
    if resource.get("resourceType") != "Patient":
        issues.append("resourceType must be Patient")
    _require(resource, "id", issues)
    _check_meta(resource, issues)
    if issues:
        raise FhirValidationError(issues)


def validate_encounter(resource: dict[str, Any]) -> datetime:
    issues: list[str] = []
    if resource.get("resourceType") != "Encounter":
        issues.append("resourceType must be Encounter")
    _require(resource, "id", issues)
    _require(resource, "status", issues)
    _check_meta(resource, issues)
    _check_reference(resource, "subject", "Patient", issues, required=True)
    period = resource.get("period")
    start = None
    if not isinstance(period, dict) or "start" not in period:
        issues.append("Encounter.period.start is required for the demo timeline origin")
    else:
        try:
            start = parse_instant(period.get("start"), "Encounter.period.start")
        except FhirValidationError as exc:
            issues.extend(exc.issues)
    if issues:
        raise FhirValidationError(issues)
    assert start is not None
    return start


def validate_observation(resource: dict[str, Any]) -> EffectiveTime:
    issues: list[str] = []
    if resource.get("resourceType") != "Observation":
        issues.append("resourceType must be Observation")
    _require(resource, "id", issues)
    _check_meta(resource, issues)
    status = _require(resource, "status", issues)
    if status is not None and status not in OBSERVATION_STATUS:
        issues.append(f"status '{status}' is not in the ObservationStatus value set")
    code = resource.get("code")
    codings = code.get("coding") if isinstance(code, dict) else None
    if not isinstance(codings, list) or not codings or not all(
        isinstance(c, dict) and isinstance(c.get("system"), str) and isinstance(c.get("code"), str)
        for c in codings
    ):
        issues.append("Observation.code.coding[] needs at least one coding with system and code")
    categories = resource.get("category")
    if categories is not None:
        if not isinstance(categories, list) or not all(
            isinstance(cat, dict) and isinstance(cat.get("coding"), list) for cat in categories
        ):
            issues.append("Observation.category must be a list of CodeableConcept")
    _check_reference(resource, "subject", "Patient", issues, required=True)
    _check_reference(resource, "encounter", "Encounter", issues, required=False)
    has_value = any(key.startswith("value") for key in resource)
    if has_value and resource.get("dataAbsentReason") is not None:
        issues.append("obs-6: dataAbsentReason SHALL only be present if Observation.value[x] is not present")
    if "valueQuantity" in resource and not isinstance(resource["valueQuantity"], dict):
        issues.append("valueQuantity must be a Quantity object")
    if "component" in resource:
        components = resource["component"]
        if not isinstance(components, list):
            issues.append("component must be a list")
        else:
            code_set = {(c.get("system"), c.get("code")) for c in (codings or []) if isinstance(c, dict)}
            for component in components:
                ccode = component.get("code") if isinstance(component, dict) else None
                ccodings = ccode.get("coding") if isinstance(ccode, dict) else None
                if not isinstance(ccodings, list) or not ccodings:
                    issues.append("component.code is required")
                    continue
                if has_value and any(
                    (c.get("system"), c.get("code")) in code_set for c in ccodings if isinstance(c, dict)
                ):
                    issues.append("obs-7: component.code equal to Observation.code must not carry a top-level value")
    if resource.get("issued") is not None:
        try:
            parse_instant(resource.get("issued"), "Observation.issued")
        except FhirValidationError as exc:
            issues.extend(exc.issues)
    effective: EffectiveTime | None = None
    if "effectiveDateTime" in resource and "effectivePeriod" in resource:
        issues.append("effective[x] must be a single choice")
    elif "effectiveDateTime" in resource:
        try:
            at = parse_instant(resource["effectiveDateTime"], "effectiveDateTime")
            effective = EffectiveTime(at, None, None)
        except FhirValidationError as exc:
            issues.extend(exc.issues)
    elif "effectivePeriod" in resource:
        period = resource["effectivePeriod"]
        try:
            start = parse_instant((period or {}).get("start"), "effectivePeriod.start")
            end = parse_instant((period or {}).get("end"), "effectivePeriod.end")
            if end < start:
                issues.append("effectivePeriod.end precedes effectivePeriod.start")
            else:
                effective = EffectiveTime(end, start, (end - start).total_seconds() / 3600.0)
        except FhirValidationError as exc:
            issues.extend(exc.issues)
    else:
        issues.append("effectiveDateTime or effectivePeriod is required for a timeline observation")
    if issues:
        raise FhirValidationError(issues)
    assert effective is not None
    return effective


def validate_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[str] = []
    if bundle.get("resourceType") != "Bundle":
        issues.append("resourceType must be Bundle")
    if bundle.get("type") not in BUNDLE_TYPES:
        issues.append("Bundle.type is not a valid bundle type")
    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        issues.append("Bundle.entry must be a non-empty list")
        raise FhirValidationError(issues)
    resources = []
    for index, entry in enumerate(entries):
        resource = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(resource, dict) or not isinstance(resource.get("resourceType"), str):
            issues.append(f"entry[{index}].resource is missing a resourceType")
            continue
        resources.append(resource)
    if issues:
        raise FhirValidationError(issues)
    return resources


def category_codes(resource: dict[str, Any]) -> list[str]:
    out = []
    for category in resource.get("category", []) or []:
        for coding in category.get("coding", []) or []:
            if coding.get("system") == CATEGORY_SYSTEM and isinstance(coding.get("code"), str):
                out.append(coding["code"])
    return out

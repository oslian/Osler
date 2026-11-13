"""Validate -> immutable raw ledger -> mapping registry -> canonical rows.

Idempotent on (source_system, resource_type, resource_id, version_id). A
corrected resource (new ``meta.versionId``) is a new ledger row; the earlier
version is retained. Unmapped codes/units and data-absent values are kept in
the ledger and recorded as mapping failures; they never reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import fhir, mapping
from .repository import Repository, ResourceVersionConflict


@dataclass(frozen=True)
class IngestResult:
    accepted: bool
    raw_id: int | None
    resource_type: str
    resource_id: str | None
    version_id: str | None
    ledger_duplicate: bool = False
    validation_issues: tuple[str, ...] = ()
    canonical: tuple[dict[str, Any], ...] = ()
    mapping_failure: str | None = None
    mapping_detail: str = ""
    effective_time: str | None = None
    patient_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    version_conflict: bool = False   # refused: this id and version already hold different content


class Ingestor:
    def __init__(self, repo: Repository, source_system: str):
        self.repo = repo
        self.source_system = source_system

    def ingest_bundle(self, bundle: dict[str, Any], received_at: str, event_id: str) -> list[IngestResult]:
        resources = fhir.validate_bundle(bundle)
        results = []
        for resource in resources:
            results.append(self.ingest_resource(resource, received_at=received_at, event_id=event_id))
        return results

    def ingest_resource(self, resource: dict[str, Any], *, received_at: str, event_id: str) -> IngestResult:
        resource_type = str(resource.get("resourceType"))
        try:
            if resource_type == "Patient":
                return self._ingest_patient(resource, received_at, event_id)
            if resource_type == "Encounter":
                return self._ingest_encounter(resource, received_at, event_id)
            if resource_type == "Observation":
                return self._ingest_observation(resource, received_at, event_id)
        except ResourceVersionConflict as exc:
            # Fail closed: nothing of this resource enters the ledger or the
            # timeline, and the caller is told exactly which identity clashed.
            return IngestResult(False, None, resource_type, exc.resource_id, exc.version_id,
                                validation_issues=(str(exc),), version_conflict=True,
                                patient_id=fhir.reference_id(resource, "subject") if resource_type != "Patient" else resource.get("id"))
        return IngestResult(False, None, resource_type, resource.get("id"), None,
                            validation_issues=(f"unsupported resourceType {resource_type}",))

    def _ingest_patient(self, resource: dict[str, Any], received_at: str, event_id: str) -> IngestResult:
        try:
            fhir.validate_patient(resource)
        except fhir.FhirValidationError as exc:
            return IngestResult(False, None, "Patient", resource.get("id"), None, validation_issues=tuple(exc.issues))
        meta = resource["meta"]
        with self.repo.transaction():
            insert = self.repo.insert_raw(
                source_system=self.source_system, resource_type="Patient", resource_id=resource["id"],
                version_id=meta["versionId"], patient_id=resource["id"], source_last_updated=meta["lastUpdated"],
                received_at=received_at, event_id=event_id, resource=resource, validation_status="valid")
            self.repo.upsert_patient(resource["id"], self.source_system, resource, received_at)
        return IngestResult(True, insert.raw_id, "Patient", resource["id"], meta["versionId"],
                            ledger_duplicate=not insert.created, patient_id=resource["id"])

    def _ingest_encounter(self, resource: dict[str, Any], received_at: str, event_id: str) -> IngestResult:
        try:
            start = fhir.validate_encounter(resource)
        except fhir.FhirValidationError as exc:
            return IngestResult(False, None, "Encounter", resource.get("id"), None,
                                validation_issues=tuple(exc.issues))
        meta = resource["meta"]
        patient_id = fhir.reference_id(resource, "subject")
        with self.repo.transaction():
            insert = self.repo.insert_raw(
                source_system=self.source_system, resource_type="Encounter", resource_id=resource["id"],
                version_id=meta["versionId"], patient_id=patient_id, source_last_updated=meta["lastUpdated"],
                received_at=received_at, event_id=event_id, resource=resource, validation_status="valid")
            self.repo.upsert_encounter(resource["id"], patient_id, fhir.iso(start), resource, received_at)
        return IngestResult(True, insert.raw_id, "Encounter", resource["id"], meta["versionId"],
                            ledger_duplicate=not insert.created, patient_id=patient_id,
                            extra={"period_start": fhir.iso(start)})

    def _ingest_observation(self, resource: dict[str, Any], received_at: str, event_id: str) -> IngestResult:
        try:
            effective = fhir.validate_observation(resource)
        except fhir.FhirValidationError as exc:
            return IngestResult(False, None, "Observation", resource.get("id"), None,
                                validation_issues=tuple(exc.issues))
        meta = resource["meta"]
        patient_id = fhir.reference_id(resource, "subject")
        effective_iso = fhir.iso(effective.effective)
        with self.repo.transaction():
            insert = self.repo.insert_raw(
                source_system=self.source_system, resource_type="Observation", resource_id=resource["id"],
                version_id=meta["versionId"], patient_id=patient_id, source_last_updated=meta["lastUpdated"],
                received_at=received_at, event_id=event_id, resource=resource, validation_status="valid")
            if not insert.created:
                canonical = tuple(self.repo.canonical_for_raw(insert.raw_id))
                return IngestResult(True, insert.raw_id, "Observation", resource["id"], meta["versionId"],
                                    ledger_duplicate=True, canonical=canonical, effective_time=effective_iso,
                                    patient_id=patient_id)
            outcome = mapping.map_observation(resource, effective.effective, effective.period_hours)
            if not outcome.mapped:
                self.repo.insert_mapping_failure(insert.raw_id, patient_id, outcome.failure or "unmapped",
                                                 outcome.detail, received_at)
                return IngestResult(True, insert.raw_id, "Observation", resource["id"], meta["versionId"],
                                    mapping_failure=outcome.failure, mapping_detail=outcome.detail,
                                    effective_time=effective_iso, patient_id=patient_id)
            canonical_rows = []
            for value in outcome.values:
                canonical_id = self.repo.insert_canonical(
                    insert.raw_id, patient_id, resource["id"], meta["versionId"], resource["status"],
                    variable=value.variable, value=float(value.value), unit=value.unit,
                    effective_time=fhir.iso(value.effective_time), issued=resource.get("issued"),
                    source_last_updated=meta["lastUpdated"], received_at=received_at,
                    mapping_version=mapping.MAPPING_VERSION, rule_key="|".join(value.rule_key),
                    input_value=float(value.input_value), input_unit=value.input_unit,
                    derivation=value.derivation)
                canonical_rows.append({
                    "canonical_id": canonical_id, "variable": value.variable, "value": float(value.value),
                    "unit": value.unit, "effective_time": fhir.iso(value.effective_time),
                    "input_value": float(value.input_value), "input_unit": value.input_unit,
                    "derivation": value.derivation, "rule_key": "|".join(value.rule_key),
                    "mapping_version": mapping.MAPPING_VERSION,
                })
        return IngestResult(True, insert.raw_id, "Observation", resource["id"], meta["versionId"],
                            canonical=tuple(canonical_rows), effective_time=effective_iso, patient_id=patient_id)

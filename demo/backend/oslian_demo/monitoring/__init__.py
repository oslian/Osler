"""Oslian synthetic FHIR R4 streaming demo.

Boundaries (one responsibility per module):

    source.py      ClinicalDataSource: MockFhirSource now, EpicFhirSource contract only
    fhir.py        FHIR R4 structural validation + timestamp helpers
    mapping.py     (code system, code, shape, unit) -> canonical variable registry
    repository.py  SQLite (WAL) append-only raw ledger, canonical rows, sessions, runs
    ingest.py      validate -> ledger -> map -> canonical, idempotent
    timeline.py    delivered-resource timeline, watermark, model input (no future leak)
    forecast_runtime.py  ForecastRuntime adapter (detached, or the forecast API over HTTP)
    session.py     stream playback clock, event bus, coalesced run scheduler
    anatomy.py     organ <-> variable/cell mapping loaded from the approved file
    api.py         Flask blueprint: REST control + SSE
"""

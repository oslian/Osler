"""Oslian Demo — a research-only demonstration of two independent workflows.

    medication/   symbolic medication-safety candidate ranking (deterministic
                  engine; the LLM only parses free text and explains the result)
    monitoring/   synthetic FHIR R4 stream + retrospective replay, ledgered in
                  SQLite and forecast by the WholeBody runtime adapter

Nothing here diagnoses, prescribes, raises a clinical alert, or estimates a
causal treatment effect.
"""

__all__ = ["create_app"]

PRODUCT_NAME = "Oslian Demo"


def create_app(*args, **kwargs):
    """Lazy re-export so ``import oslian_demo`` does not require Flask."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)

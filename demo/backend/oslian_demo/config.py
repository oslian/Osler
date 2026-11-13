"""Every path and environment variable the demo reads, in one place.

Paths fall into three kinds and are kept apart deliberately:

*Package data* ships with the backend and is addressed relative to this file,
so a checkout runs with no configuration at all: the knowledge JSONs, the stream
fixtures and the approved organ/function maps.

*Frontend* is a sibling tree, not package data. Flask serves ``frontend/`` as
its static folder and its template folder, so the browser assets stay in one
obvious place rather than buried inside the Python package.

*Runtime state* is machine-local and must never be committed: the SQLite ledger
and anything else written while the demo runs. It defaults to ``var/`` beside
the repository root and is overridable by environment variable.

Environment variables (all optional; see ``.env.example``):

    OSLIAN_HOST                     bind address                (127.0.0.1)
    OSLIAN_PORT                     HTTP port                   (5000)
    OSLIAN_STREAM_DB                SQLite ledger path          (var/oslian_stream.sqlite3)
    OSLIAN_STREAM_LOG               events|off                  (events)
    OSLIAN_STREAM_LOG_FORMAT        pretty|json                 (pretty)
    OSLIAN_STREAM_LOG_PAYLOAD       summary|full                (summary)
    OSLIAN_WHOLEBODY_RUNTIME        fixture|release             (fixture)
    OSLIAN_WHOLEBODY_RELEASE_DIR    release checkout            (backend/research/wholebody-jepa)
    OSLIAN_WHOLEBODY_PYTHON         interpreter for the release worker
    OSLIAN_WHOLEBODY_PYTHONPATH     dependency overlay for that interpreter
    OSLIAN_EPIC_PRIVATE_KEY_PEM     referenced by name only; never read here
    LLM_PROVIDER                    openai|gemini               (openai)
    OPENAI_API_KEY / OPENAI_MODEL   Medication Safety chat and free-text parsing
    GEMINI_API_KEY / GEMINI_MODEL   the same, for the Gemini provider
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

# ---- repository layout -----------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent          # backend/oslian_demo
BACKEND_DIR = PACKAGE_DIR.parent                       # backend
REPO_ROOT = BACKEND_DIR.parent                         # the checkout root

FRONTEND_DIR = REPO_ROOT / "frontend"
CSS_DIR = FRONTEND_DIR / "css"
JS_DIR = FRONTEND_DIR / "js"
ANATOMY_ASSETS_DIR = FRONTEND_DIR / "assets" / "anatomy-motion"

MEDICATION_DIR = PACKAGE_DIR / "medication"
MEDICATION_DATA_DIR = MEDICATION_DIR / "data"

MONITORING_DIR = PACKAGE_DIR / "monitoring"
FIXTURE_DIR = MONITORING_DIR / "fixtures"
ORGAN_MAP_PATH = MONITORING_DIR / "organ_target_map.json"
FUNCTION_MAP_PATH = MONITORING_DIR / "functional_group_map_v1.json"
RELEASE_CONTRACT_FIXTURE = FIXTURE_DIR / "wholebody_release_contract_v2.json"

# ---- the WholeBody research submodule and its lock -------------------------
RESEARCH_DIR = BACKEND_DIR / "research" / "oslian-model"
MODEL_LOCK_PATH = BACKEND_DIR / "model-lock.json"

# ---- runtime state ---------------------------------------------------------
VAR_DIR = REPO_ROOT / "var"
DEFAULT_STREAM_DB = VAR_DIR / "oslian_stream.sqlite3"


def load_model_lock(path: Path = MODEL_LOCK_PATH) -> dict:
    """The approved model release. The release runtime may load nothing else."""
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != "oslian.model-lock.v1":
        raise ValueError(f"unsupported model lock schema in {path}")
    return lock


@dataclass(frozen=True)
class AppConfig:
    """Deployment settings, read from the environment exactly once."""

    host: str
    port: int
    stream_db: Path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AppConfig":
        env = os.environ if env is None else env
        db = env.get("OSLIAN_STREAM_DB") or DEFAULT_STREAM_DB
        return cls(host=env.get("OSLIAN_HOST", "127.0.0.1"),
                   port=int(env.get("OSLIAN_PORT", "5000")),
                   stream_db=Path(db))

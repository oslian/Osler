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
    OSLIAN_FORECAST_RUNTIME         detached|api                (detached)
    OSLIAN_FORECAST_API_URL         the forecast API in api/    (http://127.0.0.1:8000)
    OSLIAN_FORECAST_API_TOKEN       its OSLER_API_TOKEN; required, with https, unless the API is on this computer
    OSLIAN_AUTH                     off|oidc                    (off; off binds to loopback only)
    OSLIAN_PUBLIC_URL               the https origin users open (required with oidc)
    OSLIAN_OIDC_ISSUER              the identity provider's issuer URL (required with oidc)
    OSLIAN_OIDC_CLIENT_ID / OSLIAN_OIDC_CLIENT_SECRET          (required with oidc)
    OSLIAN_OIDC_REQUIRED_GROUP      only users whose "groups" claim holds this may sign in  (anyone the provider admits)
    OSLIAN_SESSION_SECRET           signs the session cookie, >= 32 characters (required with oidc)
    OSLIAN_IDLE_MINUTES             automatic logoff after this long idle      (15)
    OSLIAN_AUDIT_DB                 SQLite access audit ledger  (var/oslian_audit.sqlite3)
    OSLIAN_TRUSTED_PROXY            address of the TLS proxy in front; only it may name the client's address and scheme  (none)
    OSLIAN_EXTERNAL_LLM             on|off: may text leave for OpenAI/Gemini   (on without auth, off with oidc)
    OSLIAN_EPIC_PRIVATE_KEY_PEM     referenced by name only; never read here
    LLM_PROVIDER                    openai|gemini               (openai)
    OPENAI_API_KEY / OPENAI_MODEL   Medication Safety chat and free-text parsing
    GEMINI_API_KEY / GEMINI_MODEL   the same, for the Gemini provider
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

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
FORECAST_CONTRACT_FIXTURE = FIXTURE_DIR / "forecast_api_contract_v1.json"

# ---- runtime state ---------------------------------------------------------
VAR_DIR = REPO_ROOT / "var"
DEFAULT_STREAM_DB = VAR_DIR / "oslian_stream.sqlite3"
DEFAULT_AUDIT_DB = VAR_DIR / "oslian_audit.sqlite3"
SESSION_HOURS = 12                                     # sign in again at least once a shift
SERVER_THREADS = 32                                    # each open event stream holds one for as long as the page is open


def is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def external_llm_allowed(env: dict[str, str] | None = None) -> bool:
    """May request text be sent to OpenAI or Gemini?  With sign-in on, only if the deployment says so: text typed about a
    patient is protected health information, and it may go only to a vendor under a business associate agreement."""

    env = os.environ if env is None else env
    default = "on" if env.get("OSLIAN_AUTH", "off") == "off" else "off"
    return env.get("OSLIAN_EXTERNAL_LLM", default) == "on"


@dataclass(frozen=True)
class AppConfig:
    """Deployment settings, read from the environment exactly once.

    Fail closed: without sign-in the server binds to loopback only, and sign-in refuses to start half-configured.
    """

    host: str
    port: int
    stream_db: Path
    auth: str = "off"
    audit_db: Path = DEFAULT_AUDIT_DB
    idle_minutes: int = 15
    public_url: str = ""
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    session_secret: str = ""
    required_group: str = ""
    trusted_proxy: str = ""

    def __post_init__(self) -> None:
        if self.auth not in ("off", "oidc"):
            raise ValueError(f"OSLIAN_AUTH must be off or oidc, not {self.auth!r}")
        if self.idle_minutes < 1:
            raise ValueError("OSLIAN_IDLE_MINUTES must be at least 1")
        if self.auth == "off":
            if not is_loopback(self.host):
                raise ValueError(f"OSLIAN_HOST={self.host} is reachable from other machines; set OSLIAN_AUTH=oidc or bind to 127.0.0.1")
            return
        missing = [name for name, value in (("OSLIAN_PUBLIC_URL", self.public_url), ("OSLIAN_OIDC_ISSUER", self.oidc_issuer),
                                            ("OSLIAN_OIDC_CLIENT_ID", self.oidc_client_id),
                                            ("OSLIAN_OIDC_CLIENT_SECRET", self.oidc_client_secret),
                                            ("OSLIAN_SESSION_SECRET", self.session_secret)) if not value]
        if missing:
            raise ValueError(f"OSLIAN_AUTH=oidc needs {', '.join(missing)}")
        if len(self.session_secret) < 32:
            raise ValueError("OSLIAN_SESSION_SECRET must be at least 32 characters")
        for name, url in (("OSLIAN_PUBLIC_URL", self.public_url), ("OSLIAN_OIDC_ISSUER", self.oidc_issuer)):
            parts = urlsplit(url)
            if parts.scheme != "https" and not (parts.scheme == "http" and is_loopback(parts.hostname or "")):
                raise ValueError(f"{name} must be an https URL")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AppConfig":
        env = os.environ if env is None else env
        if env.get("OSLIAN_AUTH", "off") != "off" and env.get("OSLIAN_STREAM_LOG_PAYLOAD", "summary") == "full":
            raise ValueError("OSLIAN_STREAM_LOG_PAYLOAD=full would print patient resources to the console; use summary with sign-in")
        stream_db = env.get("OSLIAN_STREAM_DB") or DEFAULT_STREAM_DB
        # a ledger kept in memory (the tests) gets an audit ledger in memory too
        audit_db = env.get("OSLIAN_AUDIT_DB") or (":memory:" if str(stream_db) == ":memory:" else DEFAULT_AUDIT_DB)
        return cls(host=env.get("OSLIAN_HOST", "127.0.0.1"),
                   port=int(env.get("OSLIAN_PORT", "5000")),
                   stream_db=Path(stream_db),
                   auth=env.get("OSLIAN_AUTH", "off"),
                   audit_db=Path(audit_db),
                   idle_minutes=int(env.get("OSLIAN_IDLE_MINUTES", "15")),
                   public_url=env.get("OSLIAN_PUBLIC_URL", "").rstrip("/"),
                   oidc_issuer=env.get("OSLIAN_OIDC_ISSUER", "").rstrip("/"),
                   oidc_client_id=env.get("OSLIAN_OIDC_CLIENT_ID", ""),
                   oidc_client_secret=env.get("OSLIAN_OIDC_CLIENT_SECRET", ""),
                   session_secret=env.get("OSLIAN_SESSION_SECRET", ""),
                   required_group=env.get("OSLIAN_OIDC_REQUIRED_GROUP", ""),
                   trusted_proxy=env.get("OSLIAN_TRUSTED_PROXY", ""))

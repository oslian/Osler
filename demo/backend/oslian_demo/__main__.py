"""``python -m oslian_demo`` — start the demo server."""

from __future__ import annotations

from waitress import serve

from .app import create_app
from .config import SERVER_THREADS, AppConfig
from .medication.routes import DRUGS_PKPD, SAMPLE_CASES
from .medication import llm_client


def server_options(config: AppConfig) -> dict:
    """What Waitress is started with. Only the proxy named in ``OSLIAN_TRUSTED_PROXY`` may say which address a request
    came from and whether it arrived over https; from anyone else those headers are dropped."""

    options = {"host": config.host, "port": config.port, "threads": SERVER_THREADS, "ident": None}
    if config.trusted_proxy:
        options.update(trusted_proxy=config.trusted_proxy, trusted_proxy_count=1,
                       trusted_proxy_headers={"x-forwarded-for", "x-forwarded-proto"}, clear_untrusted_proxy_headers=True)
    return options


def main() -> None:
    config = AppConfig.from_env()
    app = create_app(config)
    service = app.extensions["oslian_stream"]

    print(f"[oslian-demo] drugs={len(DRUGS_PKPD)} cases={len(SAMPLE_CASES)} "
          f"env_llm={'on' if llm_client.available() else 'off'} "
          f"runtime={service.runtime.config.mode} db={config.stream_db}")
    print(f"[oslian-demo] http://{config.host}:{config.port}")
    # Waitress is a production WSGI server; it speaks plain HTTP, so TLS is the proxy's. One thread per request, so an
    # open event stream never blocks control requests. It returns on Ctrl+C.
    try:
        serve(app, **server_options(config))
    finally:
        # Ctrl+C must leave nothing behind: this cancels any pending run, waits
        # only for a run already executing, and flushes the console writer.
        service.close()
        print("[oslian-demo] stopped")


if __name__ == "__main__":
    main()

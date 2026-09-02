"""``python -m oslian_demo`` — start the demo server."""

from __future__ import annotations

import logging

from .app import create_app
from .config import AppConfig
from .medication.routes import DRUGS_PKPD, SAMPLE_CASES
from .medication import llm_client


def main() -> None:
    config = AppConfig.from_env()
    app = create_app(config)
    service = app.extensions["oslian_stream"]

    # The demo's console is the two-stage event log, not one line per asset
    # request. Werkzeug's access log writes from the request thread, so leaving
    # it on makes every response wait on terminal throughput; warnings and
    # errors still come through.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"[oslian-demo] drugs={len(DRUGS_PKPD)} cases={len(SAMPLE_CASES)} "
          f"env_llm={'on' if llm_client.available() else 'off'} "
          f"runtime={service.runtime.config.mode} db={config.stream_db}")
    print(f"[oslian-demo] http://{config.host}:{config.port}")
    # threaded so SSE subscribers never block control requests; no reloader so
    # the playback threads are not duplicated by a second interpreter.
    try:
        app.run(host=config.host, port=config.port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("[oslian-demo] stopping")
    finally:
        # Ctrl+C must leave nothing behind: this cancels any pending run, waits
        # only for a run already executing, flushes the console writer and shuts
        # the release worker down through its own protocol rather than orphaning
        # the subprocess.
        service.close()


if __name__ == "__main__":
    main()

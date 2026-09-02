"""``python -m oslian_demo`` — start the demo server."""

from __future__ import annotations

from .app import create_app
from .config import AppConfig
from .medication.routes import DRUGS_PKPD, SAMPLE_CASES
from .medication import llm_client


def main() -> None:
    config = AppConfig.from_env()
    app = create_app(config)
    service = app.extensions["oslian_stream"]
    print(f"[oslian-demo] drugs={len(DRUGS_PKPD)} cases={len(SAMPLE_CASES)} "
          f"env_llm={'on' if llm_client.available() else 'off'} "
          f"runtime={service.runtime.config.mode} db={config.stream_db}")
    print(f"[oslian-demo] http://{config.host}:{config.port}")
    # threaded so SSE subscribers never block control requests; no reloader so
    # the playback threads are not duplicated by a second interpreter.
    app.run(host=config.host, port=config.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

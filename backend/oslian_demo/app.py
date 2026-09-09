"""The Flask application factory: the single entry point of Oslian Demo.

Two independent workflows are mounted as separate blueprints and share nothing
but this app object:

    medication.routes   Medication Safety — symbolic candidate ranking
    monitoring.api      Monitoring — synthetic FHIR R4 stream and replay
    monitoring.ward     Ward attention order — observations only, no runtime

The browser assets are not package data: Flask serves the sibling ``frontend/``
tree as both its static folder and its template folder, so the front end stays
one obvious directory rather than something buried inside the Python package.

Run it with ``python -m oslian_demo`` (or ``scripts/start-demo.ps1``).
"""

from __future__ import annotations

from flask import Flask, render_template

from .config import AppConfig, FRONTEND_DIR
from .medication import routes as medication_routes
from .monitoring.api import StreamService, create_blueprint as create_monitoring_blueprint
from .monitoring.ward import create_blueprint as create_ward_blueprint


def create_app(config: AppConfig | None = None, stream_service: StreamService | None = None) -> Flask:
    config = config or AppConfig.from_env()
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static",
                template_folder=str(FRONTEND_DIR))
    # demo assets change often; make the browser revalidate static files on every load
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["OSLIAN"] = config

    service = stream_service or StreamService(db_path=config.stream_db)
    app.extensions["oslian_stream"] = service

    app.register_blueprint(medication_routes.bp)
    app.register_blueprint(create_monitoring_blueprint(service))
    app.register_blueprint(create_ward_blueprint())

    @app.get("/")
    def index():
        return render_template("index.html")

    return app

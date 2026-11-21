"""Access control: sign-in through the hospital's identity provider, automatic logoff, response headers, and the audit
hook that records every API request.

Sign-in is OpenID Connect (Authlib), so the hospital's own directory decides who a user is and how they prove it
(password policy, multi-factor); this application stores no credential. The session is a signed cookie holding the
user's subject, display name and two timestamps, nothing clinical.

With ``OSLIAN_AUTH=off`` nothing is asked of the user and the actor is ``local-demo``; ``AppConfig`` allows that on
loopback only. The audit ledger is written in both modes.

Authorisation is one level: a signed-in user sees every patient the server holds. ``OSLIAN_OIDC_REQUIRED_GROUP`` narrows
who may sign in to one group of the provider's directory. Roles and per-patient access are not implemented.

Every page carries a Content Security Policy: scripts run only from this site, from the two pinned library files, or
with the response's nonce.

With sign-in on, requests are rate limited (Flask-Limiter): per user for the API, per address for the sign-in routes.
The counters live in this process's memory, so they hold for one server process.
"""

from __future__ import annotations

from datetime import timedelta
import secrets
import time

from flask import Flask, g, jsonify, redirect, request, session

from ..config import SESSION_HOURS, AppConfig
from .audit import AuditLog

LOCAL_ACTOR = "local-demo"
OPEN_ENDPOINTS = frozenset({"access_login", "access_callback", "access_logout", "static"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
API_LIMIT = "600 per minute"                                # one user's page polls a few times a second at most
SIGN_IN_LIMIT = "20 per minute"                             # per address, on the routes open before sign-in
# The page's own scripts carry the nonce; marked and vis-network come from two CDNs, pinned by integrity hash in
# index.html. Inline style attributes are used throughout the page, so styles stay 'unsafe-inline'.
CONTENT_SECURITY_POLICY = ("default-src 'self'; script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net https://unpkg.com; "
                           "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
                           "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
                           "form-action 'self'; frame-ancestors 'none'")
SIGNED_OUT_PAGE = ('<!doctype html><meta charset="utf-8"><title>Signed out</title>'
                   '<p style="font:16px sans-serif;margin:3rem">You are signed out. <a href="/login">Sign in</a></p>')


def init_access(app: Flask, config: AppConfig, audit: AuditLog) -> None:
    signed_in = config.auth == "oidc"
    idle_seconds = config.idle_minutes * 60

    if signed_in:
        from authlib.integrations.flask_client import OAuth   # only a deployment with sign-in needs these two
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        limiter = Limiter(key_func=lambda: g.get("actor") or get_remote_address(), default_limits=[API_LIMIT],
                          storage_uri="memory://")
        app.secret_key = config.session_secret
        app.config.update(SESSION_COOKIE_NAME="oslian_session", SESSION_COOKIE_SECURE=config.public_url.startswith("https://"),
                          SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                          PERMANENT_SESSION_LIFETIME=timedelta(seconds=idle_seconds))
        oauth = OAuth(app)
        oauth.register("idp", server_metadata_url=f"{config.oidc_issuer}/.well-known/openid-configuration",
                       client_id=config.oidc_client_id, client_secret=config.oidc_client_secret,
                       client_kwargs={"scope": "openid profile email"})

        @app.get("/login", endpoint="access_login")
        @limiter.limit(SIGN_IN_LIMIT, key_func=get_remote_address)
        def login():
            return oauth.idp.authorize_redirect(f"{config.public_url}/auth/callback")

        @app.get("/auth/callback", endpoint="access_callback")
        @limiter.limit(SIGN_IN_LIMIT, key_func=get_remote_address)
        def callback():
            claims = oauth.idp.authorize_access_token().get("userinfo") or {}
            if not claims.get("sub"):
                return jsonify({"error": "the identity provider returned no subject"}), 403
            g.actor, g.actor_name = str(claims["sub"]), claims.get("name")
            if config.required_group and config.required_group not in (claims.get("groups") or []):
                return jsonify({"error": "this account is not in the group allowed to use Oslian"}), 403
            now = time.time()
            session.clear()
            session.permanent = True
            session.update(sub=str(claims["sub"]), name=claims.get("name") or claims.get("email") or str(claims["sub"]),
                           login_at=now, seen_at=now)
            g.actor, g.actor_name = session["sub"], session["name"]
            return redirect("/")

        @app.get("/logout", endpoint="access_logout")
        @limiter.limit(SIGN_IN_LIMIT, key_func=get_remote_address)
        def logout():
            g.actor, g.actor_name = session.get("sub"), session.get("name")
            session.clear()
            return SIGNED_OUT_PAGE

    @app.before_request
    def require_user():
        g.actor, g.actor_name = (None, None) if signed_in else (LOCAL_ACTOR, None)
        g.csp_nonce = secrets.token_urlsafe(16)
        if not signed_in or request.endpoint in OPEN_ENDPOINTS:
            return None
        # A browser names the page that sent a state-changing request; another site's page may not send ours.
        if request.method not in SAFE_METHODS and request.headers.get("Origin", config.public_url) != config.public_url:
            return jsonify({"error": "cross-site request refused"}), 403
        now = time.time()
        expired = (now - session.get("seen_at", 0) > idle_seconds or now - session.get("login_at", 0) > SESSION_HOURS * 3600)
        if not session.get("sub") or expired:
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "sign in required"}), 401
            return redirect("/login")
        session["seen_at"] = now                               # also re-issues the cookie with a fresh expiry
        g.actor, g.actor_name = session["sub"], session["name"]
        return None

    if signed_in:
        limiter.init_app(app)                                  # after require_user, so the limit is counted per user

        @app.errorhandler(429)
        def too_many(error):
            return jsonify({"error": "too many requests; wait a minute"}), 429

    @app.after_request
    def headers_and_audit(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY.format(nonce=g.get("csp_nonce", "")))
        if request.endpoint != "static":                       # pages and API answers can hold patient data
            response.headers["Cache-Control"] = "no-store"
        if signed_in and config.public_url.startswith("https://"):
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if request.path.startswith("/api/") or request.endpoint in OPEN_ENDPOINTS - {"static"}:
            audit.record(actor=g.get("actor") or "anonymous", actor_name=g.get("actor_name"), action=request.method,
                         path=request.path, patient_id=(request.view_args or {}).get("patient_id"),
                         status=response.status_code, remote_addr=request.remote_addr)
        return response

    @app.context_processor
    def page_context():
        return {"csp_nonce": g.csp_nonce, "signed_in_user": session.get("name") if signed_in else None,
                "idle_minutes": config.idle_minutes if signed_in else 0}

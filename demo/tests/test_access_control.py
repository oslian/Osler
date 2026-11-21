"""Access control, automatic logoff, the audit ledger and the fail-closed settings."""

import os
import sqlite3
import unittest
from unittest import mock

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_FORECAST_RUNTIME", "detached")

from oslian_demo.app import create_app  # noqa: E402
from oslian_demo.config import AppConfig, SESSION_HOURS, external_llm_allowed  # noqa: E402
from oslian_demo.medication import llm_client  # noqa: E402
from oslian_demo.shared.audit import AuditLog  # noqa: E402

OIDC = {"OSLIAN_STREAM_DB": ":memory:", "OSLIAN_AUTH": "oidc", "OSLIAN_PUBLIC_URL": "https://oslian.example.org",
        "OSLIAN_OIDC_ISSUER": "https://idp.example.org", "OSLIAN_OIDC_CLIENT_ID": "oslian",
        "OSLIAN_OIDC_CLIENT_SECRET": "client-secret", "OSLIAN_SESSION_SECRET": "s" * 32, "OSLIAN_IDLE_MINUTES": "15"}


class SettingsTests(unittest.TestCase):
    def test_without_sign_in_only_loopback_may_be_bound(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertEqual(AppConfig.from_env({"OSLIAN_HOST": host}).auth, "off")
        for host in ("0.0.0.0", "10.0.0.5", "oslian.example.org", "::"):
            with self.assertRaises(ValueError, msg=host):
                AppConfig.from_env({"OSLIAN_HOST": host})
        self.assertEqual(AppConfig.from_env({**OIDC, "OSLIAN_HOST": "0.0.0.0"}).auth, "oidc")

    def test_sign_in_refuses_to_start_half_configured(self):
        for name in ("OSLIAN_PUBLIC_URL", "OSLIAN_OIDC_ISSUER", "OSLIAN_OIDC_CLIENT_ID", "OSLIAN_OIDC_CLIENT_SECRET",
                     "OSLIAN_SESSION_SECRET"):
            with self.assertRaises(ValueError, msg=name):
                AppConfig.from_env({k: v for k, v in OIDC.items() if k != name})
        for bad in ({"OSLIAN_SESSION_SECRET": "short"}, {"OSLIAN_PUBLIC_URL": "http://oslian.example.org"},
                    {"OSLIAN_OIDC_ISSUER": "http://idp.example.org"}, {"OSLIAN_AUTH": "password"}, {"OSLIAN_IDLE_MINUTES": "0"}):
            with self.assertRaises(ValueError, msg=bad):
                AppConfig.from_env({**OIDC, **bad})

    def test_text_leaves_for_an_outside_model_only_when_allowed(self):
        self.assertTrue(external_llm_allowed({}))
        self.assertFalse(external_llm_allowed({"OSLIAN_AUTH": "oidc"}))
        self.assertTrue(external_llm_allowed({"OSLIAN_AUTH": "oidc", "OSLIAN_EXTERNAL_LLM": "on"}))
        self.assertFalse(external_llm_allowed({"OSLIAN_EXTERNAL_LLM": "off"}))
        with mock.patch.dict(os.environ, {"OSLIAN_EXTERNAL_LLM": "off", "OPENAI_API_KEY": "from-env"}):
            self.assertIsNone(llm_client.resolve_key("openai", "from-browser"))
            self.assertFalse(llm_client.available("from-browser"))
        with mock.patch.dict(os.environ, {"OSLIAN_EXTERNAL_LLM": "on"}):
            self.assertEqual(llm_client.resolve_key("openai", "from-browser"), "from-browser")


class AuditLedgerTests(unittest.TestCase):
    def test_rows_cannot_be_changed_or_removed(self):
        log = AuditLog(":memory:")
        log.record(actor="u1", actor_name="Nurse One", action="GET", path="/api/patients/p1/state", patient_id="p1",
                   status=200, remote_addr="127.0.0.1")
        for statement in ("UPDATE access_audit SET actor = 'someone-else'", "DELETE FROM access_audit"):
            with self.assertRaises(sqlite3.DatabaseError, msg=statement):
                log._conn.execute(statement)
        self.assertEqual([(r["actor"], r["patient_id"], r["status"]) for r in log.rows(patient_id="p1")], [("u1", "p1", 200)])
        self.assertEqual(log.rows(actor="nobody"), [])


class LocalDemoTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(AppConfig.from_env({"OSLIAN_STREAM_DB": ":memory:"}))
        self.addCleanup(self.app.extensions["oslian_stream"].close)
        self.client = self.app.test_client()

    def test_no_sign_in_is_asked_and_requests_are_still_audited(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/patients/p-404/state?session_id=x").status_code, 404)
        self.assertEqual(self.client.get("/login").status_code, 404, "no sign-in routes without sign-in")
        rows = self.app.extensions["oslian_audit"].rows()
        self.assertEqual([(r["actor"], r["action"], r["path"], r["patient_id"], r["status"]) for r in rows],
                         [("local-demo", "GET", "/api/patients/p-404/state", "p-404", 404)],
                         "pages and static files are not audited, and the query string is not kept")

    def test_headers(self):
        api = self.client.get("/api/demo-streams/sources")
        self.assertEqual(api.headers["Cache-Control"], "no-store")
        self.assertEqual(api.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(api.headers["X-Frame-Options"], "DENY")
        self.assertNotIn("Strict-Transport-Security", api.headers)
        self.assertEqual(self.client.get("/").headers["Cache-Control"], "no-store")
        self.assertIn('data-idle-minutes="0"', self.client.get("/").get_data(as_text=True))
        with self.client.get("/static/js/access.js") as script:
            self.assertNotEqual(script.headers.get("Cache-Control"), "no-store")


class SignInTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(AppConfig.from_env(OIDC))
        self.addCleanup(self.app.extensions["oslian_stream"].close)
        self.client = self.app.test_client(use_cookies=True)
        self.clock = mock.patch("oslian_demo.shared.access.time.time", return_value=1_000_000.0).start()
        self.addCleanup(mock.patch.stopall)
        self.idp = self.app.extensions["authlib.integrations.flask_client"].idp

    def sign_in(self, claims=None):
        claims = {"sub": "u-17", "name": "Nurse One"} if claims is None else claims
        with mock.patch.object(self.idp, "authorize_access_token", return_value={"userinfo": claims}):
            return self.client.get("/auth/callback", base_url="https://oslian.example.org")

    def get(self, path):
        return self.client.get(path, base_url="https://oslian.example.org")

    def audit(self):
        return [(r["actor"], r["path"], r["status"]) for r in self.app.extensions["oslian_audit"].rows()]

    def test_nothing_is_served_before_sign_in(self):
        self.assertEqual(self.get("/api/ward").status_code, 401)
        self.assertEqual(self.get("/api/demo-streams").get_json(), {"error": "sign in required"})
        page = self.get("/")
        self.assertEqual((page.status_code, page.headers["Location"]), (302, "/login"))
        with self.get("/static/js/access.js") as script:
            self.assertEqual(script.status_code, 200)
        self.assertEqual(self.audit(), [("anonymous", "/api/ward", 401), ("anonymous", "/api/demo-streams", 401)])

    def test_sign_in_goes_to_the_identity_provider_with_this_sites_callback(self):
        with mock.patch.object(self.idp, "authorize_redirect", return_value=("", 302)) as redirect:
            self.get("/login")
        redirect.assert_called_once_with("https://oslian.example.org/auth/callback")

    def test_a_signed_in_user_is_served_and_named_in_the_audit(self):
        self.assertEqual(self.sign_in().status_code, 302)
        self.assertEqual(self.get("/api/ward").status_code, 200)
        page = self.get("/").get_data(as_text=True)
        self.assertIn('data-idle-minutes="15"', page)
        self.assertIn("Signed in as Nurse One", page)
        self.assertEqual(self.audit(), [("u-17", "/auth/callback", 302), ("u-17", "/api/ward", 200)])
        cookie = self.client.get_cookie("oslian_session", domain="oslian.example.org")
        self.assertTrue(cookie.secure and cookie.http_only)
        self.assertEqual(cookie.same_site, "Lax")
        self.assertIn("Strict-Transport-Security", self.get("/api/ward").headers)

    def test_a_sign_in_without_a_subject_is_refused(self):
        self.assertEqual(self.sign_in({"name": "No Subject"}).status_code, 403)
        self.assertEqual(self.get("/api/ward").status_code, 401)

    def test_an_idle_session_ends_and_activity_keeps_it(self):
        self.sign_in()
        self.clock.return_value += 14 * 60
        self.assertEqual(self.get("/api/ward").status_code, 200)
        self.clock.return_value += 14 * 60
        self.assertEqual(self.get("/api/ward").status_code, 200, "the earlier request counted as activity")
        self.clock.return_value += 15 * 60 + 1
        self.assertEqual(self.get("/api/ward").status_code, 401)
        self.assertEqual(self.get("/api/ward").status_code, 401, "the session was cleared, not just refused once")

    def test_a_session_ends_after_a_shift_however_active(self):
        self.sign_in()
        for _ in range(SESSION_HOURS * 6):                      # a request every ten minutes, up to the limit exactly
            self.clock.return_value += 10 * 60
            self.assertEqual(self.get("/api/ward").status_code, 200)
        self.clock.return_value += 10 * 60
        self.assertEqual(self.get("/api/ward").status_code, 401)

    def test_sign_out_ends_the_session(self):
        self.sign_in()
        self.assertIn("signed out", self.get("/logout").get_data(as_text=True))
        self.assertEqual(self.get("/api/ward").status_code, 401)
        self.assertEqual(self.audit()[1], ("u-17", "/logout", 200))

    def test_a_forged_cookie_is_not_a_session(self):
        self.client.set_cookie("oslian_session", "eyJzdWIiOiJ1LTE3In0.forged.signature", domain="oslian.example.org")
        self.assertEqual(self.get("/api/ward").status_code, 401)


if __name__ == "__main__":
    unittest.main()

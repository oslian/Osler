"""Access control, automatic logoff, the audit ledger and the fail-closed settings."""

import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sqlite3
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("OSLIAN_STREAM_DB", ":memory:")
os.environ.setdefault("OSLIAN_FORECAST_RUNTIME", "detached")

from oslian_demo.app import create_app  # noqa: E402
from oslian_demo.config import AppConfig, SESSION_HOURS, external_llm_allowed  # noqa: E402
from oslian_demo.medication import llm_client  # noqa: E402
from oslian_demo.monitoring.forecast_runtime import RuntimeConfig  # noqa: E402
from oslian_demo.__main__ import server_options  # noqa: E402
from oslian_demo.shared import audit, backup  # noqa: E402
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

    def test_patient_resources_are_not_printed_to_the_console_with_sign_in(self):
        with self.assertRaises(ValueError):
            AppConfig.from_env({**OIDC, "OSLIAN_STREAM_LOG_PAYLOAD": "full"})
        AppConfig.from_env({"OSLIAN_STREAM_DB": ":memory:", "OSLIAN_STREAM_LOG_PAYLOAD": "full"})

    def test_a_forecast_api_on_another_machine_needs_tls_and_a_token(self):
        here = RuntimeConfig.from_env({"OSLIAN_FORECAST_RUNTIME": "api"})
        self.assertEqual((here.api_url, here.api_token), ("http://127.0.0.1:8000", ""))
        away = {"OSLIAN_FORECAST_RUNTIME": "api", "OSLIAN_FORECAST_API_URL": "https://forecast.example.org"}
        for bad in (away, {**away, "OSLIAN_FORECAST_API_URL": "http://forecast.example.org", "OSLIAN_FORECAST_API_TOKEN": "t"}):
            with self.assertRaises(ValueError, msg=bad):
                RuntimeConfig.from_env(bad)
        self.assertEqual(RuntimeConfig.from_env({**away, "OSLIAN_FORECAST_API_TOKEN": "t"}).api_token, "t")
        RuntimeConfig.from_env({**away, "OSLIAN_FORECAST_RUNTIME": "detached"})

    def test_only_the_named_proxy_may_say_where_a_request_came_from(self):
        plain = server_options(AppConfig.from_env({"OSLIAN_STREAM_DB": ":memory:", "OSLIAN_PORT": "5123"}))
        self.assertEqual((plain["host"], plain["port"]), ("127.0.0.1", 5123))
        self.assertNotIn("trusted_proxy", plain, "Waitress then drops forwarded headers from everyone")
        behind = server_options(AppConfig.from_env({**OIDC, "OSLIAN_TRUSTED_PROXY": "10.0.0.2"}))
        self.assertEqual((behind["trusted_proxy"], behind["trusted_proxy_count"]), ("10.0.0.2", 1))
        self.assertEqual(behind["trusted_proxy_headers"], {"x-forwarded-for", "x-forwarded-proto"})

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

    def test_export_for_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "audit.sqlite3")
            log = AuditLog(path)
            for actor, patient in (("u1", "p1"), ("u2", "p1"), ("u1", "p2")):
                log.record(actor=actor, actor_name=None, action="GET", path=f"/api/patients/{patient}/state",
                           patient_id=patient, status=200, remote_addr="127.0.0.1")
            log.close()
            out = io.StringIO()
            audit.main(["--db", path, "--patient", "p1"], out)
            lines = out.getvalue().splitlines()
            self.assertEqual(lines[0], "id,at,actor,actor_name,action,path,patient_id,status,remote_addr")
            self.assertEqual(len(lines[1].split(",")), 9, "the chain hash is not part of the export")
            self.assertEqual([line.split(",")[2] for line in lines[1:]], ["u1", "u2"])
            out = io.StringIO()
            audit.main(["--db", path, "--since", "2999-01-01"], out)
            self.assertEqual(len(out.getvalue().splitlines()), 1, "nothing that late")
            out = io.StringIO()
            audit.main(["--db", path, "--until", "2999-01-01", "--actor", "u1"], out)
            self.assertEqual(len(out.getvalue().splitlines()), 3)
            with self.assertRaises(SystemExit):
                audit.main(["--db", os.path.join(folder, "missing.sqlite3")], io.StringIO())


    def filled(self, folder, statuses=(200, 200, 403, 200)):
        path = os.path.join(folder, "audit.sqlite3")
        log = AuditLog(path)
        for index, status in enumerate(statuses):
            log.record(actor=f"u{index % 2}", actor_name=f"User {index % 2}", action="GET", path=f"/api/patients/p{index}/state",
                       patient_id=f"p{index}", status=status, remote_addr="127.0.0.1")
        log.close()
        return path

    def test_summary_for_the_regular_review(self):
        with tempfile.TemporaryDirectory() as folder:
            out = io.StringIO()
            audit.main(["--db", self.filled(folder), "--summary"], out)
            lines = [line.split(",") for line in out.getvalue().splitlines()]
            self.assertEqual(lines[0], ["actor", "actor_name", "requests", "patients", "refused", "first_at", "last_at"])
            self.assertEqual([line[:5] for line in lines[1:]], [["u0", "User 0", "2", "2", "1"], ["u1", "User 1", "2", "2", "0"]])
            out = io.StringIO()
            audit.main(["--db", self.filled(folder), "--summary", "--since", "2999-01-01"], out)
            self.assertEqual(len(out.getvalue().splitlines()), 1)

    def test_a_changed_inserted_or_removed_row_breaks_the_chain(self):
        def tamper(path, statement):
            conn = sqlite3.connect(path)
            conn.executescript("DROP TRIGGER access_audit_no_update; DROP TRIGGER access_audit_no_delete; " + statement)
            conn.close()

        with tempfile.TemporaryDirectory() as folder:
            path = self.filled(folder)
            out = io.StringIO()
            audit.main(["--db", path, "--verify"], out)
            self.assertRegex(out.getvalue(), r"^ok rows=4 head=[0-9a-f]{64}\n$")
            head = out.getvalue().split("head=")[1]
            log = AuditLog(path)                                # a restart continues the same chain
            log.record(actor="u9", actor_name=None, action="GET", path="/api/ward", patient_id=None, status=200, remote_addr=None)
            self.assertEqual(log.verify()[::2], (5, None))
            self.assertNotEqual(log.verify()[1], head.strip())
            log.close()
        for statement, bad in (("UPDATE access_audit SET actor = 'someone-else' WHERE id = 2", 2),
                               ("UPDATE access_audit SET status = 200 WHERE id = 3", 3),
                               ("DELETE FROM access_audit WHERE id = 2", 3),
                               ("INSERT INTO access_audit (at, actor, action, path, status, chain) VALUES ('2026', 'x', 'GET', '/', 200, 'made-up')", 5)):
            with tempfile.TemporaryDirectory() as folder:
                path = self.filled(folder)
                tamper(path, statement)
                with self.assertRaises(SystemExit, msg=statement) as stopped:
                    audit.main(["--db", path, "--verify"], io.StringIO())
                self.assertIn(f"row id {bad} ", str(stopped.exception), statement)


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = {"OSLIAN_STREAM_DB": os.path.join(self.folder.name, "var", "stream.sqlite3"),
                    "OSLIAN_AUDIT_DB": os.path.join(self.folder.name, "var", "audit.sqlite3")}
        app = create_app(AppConfig.from_env(self.env))
        app.test_client().get("/api/patients/p-404/state?session_id=x")
        self.assertEqual(app.test_client().get("/api/ward").status_code, 200)
        self.app = app                                          # the server stays open: a backup is taken while it runs
        self.addCleanup(app.extensions["oslian_stream"].close)
        self.addCleanup(app.extensions["oslian_audit"].close)
        self.to = os.path.join(self.folder.name, "backups")

    def back_up(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, self.env):
            backup.main(["--to", self.to], out)
        return out.getvalue().strip()

    def test_a_backup_of_the_running_ledgers_can_be_read_back(self):
        manifest = self.back_up()
        recorded = json.loads(Path(manifest).read_text())
        self.assertEqual(len(recorded["files"]), 2)
        audit_copy = next(entry for name, entry in recorded["files"].items() if name.startswith("audit-"))
        self.assertEqual(audit_copy["rows"]["access_audit"], 2)
        self.assertEqual(audit_copy["audit_head"], self.app.extensions["oslian_audit"].verify()[1])
        for name in recorded["files"]:
            self.assertEqual(os.stat(os.path.join(self.to, name)).st_mode & 0o777, 0o600)
        out = io.StringIO()
        backup.main(["--check", manifest], out)
        self.assertRegex(out.getvalue(), r"^ok 2 files from \d{8}T\d{6}Z\n$")
        restored = AuditLog(os.path.join(self.to, next(name for name in recorded["files"] if name.startswith("audit-"))))
        self.addCleanup(restored.close)
        self.assertEqual([row["path"] for row in restored.rows()], ["/api/patients/p-404/state", "/api/ward"])

    def test_a_damaged_or_missing_copy_fails_the_check(self):
        manifest = self.back_up()
        name = next(name for name in json.loads(Path(manifest).read_text())["files"] if name.startswith("audit-"))
        copy = os.path.join(self.to, name)
        with open(copy, "r+b") as handle:
            handle.seek(200)
            handle.write(b"\xff" * 16)
        with self.assertRaises(SystemExit) as stopped:
            backup.main(["--check", manifest], io.StringIO())
        self.assertIn("differs", str(stopped.exception))
        os.rename(copy, copy + ".gone")
        with self.assertRaises(SystemExit) as stopped:
            backup.main(["--check", manifest], io.StringIO())
        self.assertIn("missing", str(stopped.exception))

    def test_nothing_to_back_up(self):
        empty = {"OSLIAN_STREAM_DB": os.path.join(self.folder.name, "none.sqlite3"),
                 "OSLIAN_AUDIT_DB": os.path.join(self.folder.name, "none-audit.sqlite3")}
        with mock.patch.dict(os.environ, empty), self.assertRaises(SystemExit):
            backup.main(["--to", self.to], io.StringIO())


class IdleTimerTests(unittest.TestCase):
    """frontend/js/access.js, executed under Node with a stand-in page and clock."""

    HARNESS = """
const fs = require('fs');
let now = 0, timers = [], listeners = {}, left = null;
global.document = { body: { dataset: { idleMinutes: process.argv[2] } } };
global.window = { location: { assign: (url) => { left = url; } },
                  addEventListener: (type, fn) => { listeners[type] = fn; } };
global.setTimeout = (fn, ms) => { const t = { fn, at: now + ms, on: true }; timers.push(t); return t; };
global.clearTimeout = (t) => { if (t) t.on = false; };
const advance = (minutes) => { now += minutes * 60000;
  for (const t of timers) if (t.on && t.at <= now) { t.on = false; t.fn(); } };
eval(fs.readFileSync(process.argv[1], 'utf8'));
const out = { listening: Object.keys(listeners).sort() };
advance(14); out.after14 = left;
if (listeners.keydown) listeners.keydown();
advance(14); out.after28WithAKeyAt14 = left;
advance(1); out.after29 = left;
console.log(JSON.stringify(out));
"""

    def run_page(self, minutes):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not on PATH")
        script = os.path.join(os.path.dirname(__file__), "..", "frontend", "js", "access.js")
        done = subprocess.run([node, "-e", self.HARNESS, script, str(minutes)], capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    def test_the_page_signs_out_after_the_idle_time_and_activity_restarts_it(self):
        seen = self.run_page(15)
        self.assertEqual(seen["listening"], ["keydown", "pointerdown", "pointermove", "touchstart", "wheel"])
        self.assertEqual((seen["after14"], seen["after28WithAKeyAt14"], seen["after29"]), (None, None, "/logout"))

    def test_no_timer_without_sign_in(self):
        seen = self.run_page(0)
        self.assertEqual((seen["listening"], seen["after29"]), ([], None))


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

    def test_scripts_run_only_from_this_site_or_with_the_pages_nonce(self):
        first, second = self.client.get("/"), self.client.get("/")
        policy = first.headers["Content-Security-Policy"]
        nonce = re.search(r"script-src 'self' 'nonce-([A-Za-z0-9_-]{16,})' ", policy).group(1)
        self.assertNotIn("'unsafe-inline'", policy.split("script-src")[1].split(";")[0])
        self.assertNotIn("unsafe-eval", policy)
        for part in ("default-src 'self'", "connect-src 'self'", "object-src 'none'", "frame-ancestors 'none'"):
            self.assertIn(part, policy)
        page = first.get_data(as_text=True)
        inline = re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>", page)
        self.assertTrue(inline and all(f'nonce="{nonce}"' in attributes for attributes in inline), "every inline script has the nonce")
        self.assertNotIn(nonce, second.headers["Content-Security-Policy"], "a new nonce per response")
        outside = re.findall(r'<script[^>]*\bsrc="(https?://[^"]+)"([^>]*)>', page)
        self.assertEqual(len(outside), 2)
        for url, attributes in outside:
            self.assertRegex(url, r"@\d+\.\d+\.\d+/", "a pinned version")
            self.assertRegex(attributes, r'integrity="sha384-[A-Za-z0-9+/=]{64}" crossorigin="anonymous"')
        self.assertIn("Content-Security-Policy", self.client.get("/api/demo-streams/sources").headers)
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

    def test_only_the_allowed_group_may_sign_in(self):
        app = create_app(AppConfig.from_env({**OIDC, "OSLIAN_OIDC_REQUIRED_GROUP": "icu-clinicians"}))
        self.addCleanup(app.extensions["oslian_stream"].close)
        idp = app.extensions["authlib.integrations.flask_client"].idp
        for groups, status in ((["icu-clinicians", "staff"], 302), (["staff"], 403), (None, 403)):
            client = app.test_client(use_cookies=True)
            claims = {"sub": "u-17", "name": "Nurse One", **({"groups": groups} if groups is not None else {})}
            with mock.patch.object(idp, "authorize_access_token", return_value={"userinfo": claims}):
                self.assertEqual(client.get("/auth/callback", base_url="https://oslian.example.org").status_code, status, groups)
            served = client.get("/api/ward", base_url="https://oslian.example.org").status_code
            self.assertEqual(served, 200 if status == 302 else 401, groups)
        rows = [(r["actor"], r["path"], r["status"]) for r in app.extensions["oslian_audit"].rows()]
        self.assertIn(("u-17", "/auth/callback", 403), rows, "a refused sign-in names who tried")

    def test_another_sites_page_cannot_send_a_request_as_the_user(self):
        self.sign_in()
        post = lambda origin: self.client.post("/api/demo-streams", json={"source": "synthetic"}, base_url="https://oslian.example.org",
                                               headers={"Origin": origin} if origin else {}).status_code
        self.assertEqual(post("https://elsewhere.example"), 403)
        self.assertEqual(post("http://oslian.example.org"), 403)
        self.assertNotEqual(post("https://oslian.example.org"), 403)
        self.assertNotEqual(post(None), 403, "a server-side caller sends no Origin")
        self.assertEqual(self.client.get("/api/ward", base_url="https://oslian.example.org",
                                         headers={"Origin": "https://elsewhere.example"}).status_code, 200, "reads are not state changes")

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

    def test_requests_are_rate_limited_per_user_and_sign_in_per_address(self):
        self.sign_in()
        with mock.patch("oslian_demo.shared.access.API_LIMIT", "3 per minute"):
            app = create_app(AppConfig.from_env(OIDC))
        self.addCleanup(app.extensions["oslian_stream"].close)
        idp = app.extensions["authlib.integrations.flask_client"].idp
        clients = [app.test_client(use_cookies=True) for _ in range(2)]
        for client, sub in zip(clients, ("u-1", "u-2")):
            with mock.patch.object(idp, "authorize_access_token", return_value={"userinfo": {"sub": sub}}):
                client.get("/auth/callback", base_url="https://oslian.example.org")
        ward = lambda client: client.get("/api/ward", base_url="https://oslian.example.org")
        self.assertEqual([ward(clients[0]).status_code for _ in range(4)], [200, 200, 200, 429])
        self.assertEqual(ward(clients[0]).get_json(), {"error": "too many requests; wait a minute"})
        self.assertEqual(ward(clients[1]).status_code, 200, "another user has their own count")
        with clients[0].get("/static/js/access.js", base_url="https://oslian.example.org") as script:
            self.assertEqual(script.status_code, 200)
        rows = [(r["actor"], r["path"], r["status"]) for r in app.extensions["oslian_audit"].rows()]
        self.assertIn(("u-1", "/api/ward", 429), rows, "a refused request is audited with the user")
        statuses = [self.get("/logout").status_code for _ in range(21)]
        self.assertEqual((statuses[:20], statuses[20]), ([200] * 20, 429), "the sign-in routes are counted per address")

    def test_the_audit_names_the_client_only_on_the_proxys_word(self):
        from waitress.proxy_headers import proxy_headers_middleware
        from werkzeug.test import Client

        def remote_addr_seen(options, sender):
            app = create_app(AppConfig.from_env(OIDC))
            self.addCleanup(app.extensions["oslian_stream"].close)
            wrapped = proxy_headers_middleware(app, trusted_proxy=options.get("trusted_proxy"), clear_untrusted=True,
                                               trusted_proxy_count=options.get("trusted_proxy_count"),
                                               trusted_proxy_headers=options.get("trusted_proxy_headers"))
            Client(wrapped).get("/api/ward", environ_base={"REMOTE_ADDR": sender}, headers={"X-Forwarded-For": "203.0.113.9"})
            return app.extensions["oslian_audit"].rows()[-1]["remote_addr"]

        behind = server_options(AppConfig.from_env({**OIDC, "OSLIAN_TRUSTED_PROXY": "10.0.0.2"}))
        self.assertEqual(remote_addr_seen(behind, "10.0.0.2"), "203.0.113.9", "the proxy names the client")
        self.assertEqual(remote_addr_seen(behind, "10.0.0.66"), "10.0.0.66", "anyone else's header is dropped")
        self.assertEqual(remote_addr_seen(server_options(AppConfig.from_env(OIDC)), "10.0.0.66"), "10.0.0.66")

    def test_a_forged_cookie_is_not_a_session(self):
        self.client.set_cookie("oslian_session", "eyJzdWIiOiJ1LTE3In0.forged.signature", domain="oslian.example.org")
        self.assertEqual(self.get("/api/ward").status_code, 401)


if __name__ == "__main__":
    unittest.main()

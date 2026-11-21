# Security controls

What the demo enforces today, and what a deployment that holds real patient data still needs. The demo is
research-only and ships synthetic and de-identified data; nothing here makes it compliant with HIPAA or ready for a
SOC 2 audit. Those are properties of an organization and its operations, examined by other people, and code is one
part of them.

## What is enforced

| Control | How | HIPAA Security Rule |
|---|---|---|
| Unique user identification and authentication | With `OSLIAN_AUTH=oidc`, sign-in is OpenID Connect through the hospital's identity provider (Authlib). The provider decides password policy and multi-factor; the demo stores no credential. | §164.312(a)(2)(i), §164.312(d) |
| Nothing served before sign-in | Every page and `/api/` route needs a session. API calls get 401; pages go to `/login`. Static files (scripts, styles) are open; they hold no patient data. | §164.312(a)(1) |
| Automatic logoff | The session ends after `OSLIAN_IDLE_MINUTES` (15) without a request, and after 12 hours however active. The page also watches for pointer and keyboard activity and signs out when the user has been idle that long, because its own polling would otherwise keep the session alive. | §164.312(a)(2)(iii) |
| Audit controls | One row per API request and per sign-in, sign-out and refused request: time, user, method, path, patient id, status, remote address. Stored in `OSLIAN_AUDIT_DB`, apart from the clinical ledger. The schema rejects UPDATE and DELETE. Bodies and query strings are not recorded. | §164.312(b) |
| Fail closed | Without sign-in the server refuses to bind to anything but loopback. With sign-in it refuses to start unless every setting is present, the session secret has 32 characters and the public and provider URLs are https. | — |
| No patient text to outside vendors by default | With sign-in on, Medication Safety does not send text to OpenAI or Gemini unless `OSLIAN_EXTERNAL_LLM=on`; it falls back to rule-based parsing. Turn it on only for a vendor under a business associate agreement. | §164.308(b), §164.502(e) |
| Session cookie | Signed, `Secure`, `HttpOnly`, `SameSite=Lax`; holds the user's subject, display name and two timestamps. | §164.312(c), §164.312(e) |
| Response headers | `Cache-Control: no-store` on pages and API answers, `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and HSTS when served over https. | §164.312(e) |

Code: `backend/oslian_demo/shared/access.py`, `shared/audit.py`, `config.py`. Tests: `tests/test_access_control.py`.

## Turning sign-in on

Register the demo with the identity provider as a confidential web client with the redirect URI
`<OSLIAN_PUBLIC_URL>/auth/callback` and the scopes `openid profile email`, then set the variables under "access" in
`.env.example`. Serve it behind a TLS-terminating proxy at `OSLIAN_PUBLIC_URL`.

The round trip to a real identity provider has not been exercised: the tests replace the provider's answer.

## Not done

Technical:

- **Authorization is one level.** A signed-in user sees every patient the server holds. There are no roles, no
  per-patient or per-unit access, and no emergency ("break glass") access procedure (§164.312(a)(2)(ii)).
- **No encryption at rest in the application.** The SQLite ledgers are plain files; a deployment must put `var/` on an
  encrypted volume (§164.312(a)(2)(iv)).
- **TLS is the proxy's job.** The built-in server speaks plain HTTP and is a development server; a deployment needs a
  production WSGI server behind the proxy.
- **The forecast API in `api/` has no authentication.** It must stay on loopback or a private network beside the demo.
- **Third-party scripts and fonts load from public CDNs without integrity hashes**, and there is no Content Security
  Policy (the page uses inline scripts).
- **The console event log can print resource payloads** (`OSLIAN_STREAM_LOG_PAYLOAD=full`); keep it on `summary`.
- **No rate limiting, no CSRF token** (the cookie's `SameSite=Lax` is the only cross-site protection), no single
  sign-out at the provider.
- **Audit review and retention are not built.** Rows are kept for good and nothing reads them; HIPAA expects regular
  review (§164.308(a)(1)(ii)(D)) and six years of retention for documentation.
- **No backup, no restore test, no integrity check of the ledgers.**

Organizational (no code can supply these):

- A risk analysis and risk management plan (§164.308(a)(1)).
- Written policies, workforce training, a named security officer, sanctions.
- Business associate agreements with each hospital and with every vendor that touches patient data (hosting, any
  model provider).
- Incident response and breach notification procedures; contingency plan.
- For SOC 2: a scoped system description, the controls above operated and evidenced over time (change management,
  access reviews, vendor management, monitoring, vulnerability management), and an examination by a CPA firm. Type I
  is a point in time; Type II covers a period, usually 3 to 12 months.

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
| Sign-in limited to one group | With `OSLIAN_OIDC_REQUIRED_GROUP` set, only a user whose `groups` claim holds that value gets a session; a refused sign-in is audited with the user's subject. | §164.312(a)(1), §164.308(a)(4) |
| Content Security Policy | Scripts run only from this site, with the response's nonce, or from the two library files (marked 15.0.12, vis-network 10.1.5), which are pinned by SHA-384 integrity hash. No `unsafe-inline` or `unsafe-eval` for scripts. Connections go only to this site. | §164.312(c)(1) |
| Cross-site requests refused | With sign-in on, a state-changing request whose `Origin` is not `OSLIAN_PUBLIC_URL` gets 403. | §164.312(c)(1) |
| Forecast API token | The API in `api/` answers only callers on its own computer unless `OSLER_API_TOKEN` is set, and then only callers that send it. The demo sends `OSLIAN_FORECAST_API_TOKEN`, and refuses a forecast API on another machine without https and a token. | §164.312(d), §164.312(e) |
| Console log | With sign-in on, `OSLIAN_STREAM_LOG_PAYLOAD=full` is refused at start: it would print patient resources. | §164.312(b) |
| Audit review | `python -m oslian_demo.shared.audit --patient <id>` (or `--actor`, `--since`, `--until`) writes the matching audit rows as CSV, for example to answer who opened a patient's record. `--summary` writes one line per user for a period: requests, patients opened, refused requests. | §164.308(a)(1)(ii)(D), §164.528 |
| Audit integrity | Each audit row carries a SHA-256 over its fields and the row before it. `python -m oslian_demo.shared.audit --verify` recomputes the chain and names the first row that was changed, inserted or removed. | §164.312(c)(1), §164.312(b) |
| Backup and restore test | `python -m oslian_demo.shared.backup --to <folder>` copies both ledgers while the server runs (SQLite online backup), checks each copy, and writes a manifest with sizes, SHA-256, row counts and the audit chain's last hash. `--check <manifest>` verifies an existing backup the same way. | §164.308(a)(7)(ii)(A), §164.308(a)(7)(ii)(D) |
| Rate limits | With sign-in on: 600 requests a minute per user on the API, 20 a minute per address on the sign-in routes; over that the answer is 429, and it is audited. | — |
| Production server | `python -m oslian_demo` runs Waitress, not Flask's development server. | — |
| Client address behind the proxy | With `OSLIAN_TRUSTED_PROXY` set to the proxy's address, the audit records the address the proxy forwards; forwarded headers from any other sender are ignored. | §164.312(b) |

Code: `backend/oslian_demo/shared/access.py`, `shared/audit.py`, `shared/backup.py`, `config.py`, `__main__.py`. Tests: `tests/test_access_control.py`.

## Turning sign-in on

Register the demo with the identity provider as a confidential web client with the redirect URI
`<OSLIAN_PUBLIC_URL>/auth/callback` and the scopes `openid profile email`, then set the variables under "access" in
`.env.example`. Serve it behind a TLS-terminating proxy at `OSLIAN_PUBLIC_URL`, and set `OSLIAN_TRUSTED_PROXY` to
that proxy's address.

An audit ledger made before the hash chain was added has no `chain` column and is not read by this version; start
with a new file.

The round trip to a real identity provider has not been exercised: the tests replace the provider's answer.

## Not done

Technical:

- **Authorization is one level.** A signed-in user (in the required group, if one is set) sees every patient the
  server holds. There are no roles and no per-patient or per-unit access. Per-patient access needs the hospital's
  record of who treats whom, which arrives with an EHR launch (SMART on FHIR) and is not connected. No emergency
  ("break glass") procedure (§164.312(a)(2)(ii)).
- **No encryption at rest in the application.** The SQLite ledgers are plain files; a deployment must put `var/` and
  the API's state folder on an encrypted volume (§164.312(a)(2)(iv)).
- **TLS is the proxy's job.** Waitress speaks plain HTTP; a deployment puts a TLS-terminating proxy in front and
  sets `OSLIAN_TRUSTED_PROXY`.
- **The forecast API token is one shared secret.** There is no per-caller identity, rotation or audit ledger in the
  API. With no token it trusts any caller on its own computer, which includes a proxy running there.
- **Fonts load from Google Fonts**, so a user's browser contacts Google on each page load. Styles allow
  `unsafe-inline` because the page sets inline style attributes.
- **Rate limits are counted in one process's memory.** They reset on restart and are not shared between several
  server processes; limits against a flood belong on the proxy as well. No single sign-out at the provider.
- **Audit review is not scheduled, and nothing is alerted.** The summary and the chain check are commands a person
  (or a scheduled job the deployment sets up) has to run. Rows are never deleted; keeping six years means keeping the
  file and its backups.
- **The audit chain shows tampering inside the file only.** Someone who can rewrite the whole file can rewrite the
  chain, and rows removed from the end leave a valid shorter chain. The defense is the last hash recorded in a backup
  kept on another machine; sending it there is not automated.
- **Backups are not scheduled, not encrypted and not sent anywhere.** The command writes plain copies to a folder;
  the deployment has to put that folder on an encrypted volume on another machine and run the command on a schedule.
  The forecast API's state folder is not covered.
- **Not verified:** the round trip to a real identity provider. The browser's idle timer is tested under Node with a
  stand-in page and clock, not in a real browser with sign-in on.

Organizational (no code can supply these):

- A risk analysis and risk management plan (§164.308(a)(1)).
- Written policies, workforce training, a named security officer, sanctions.
- Business associate agreements with each hospital and with every vendor that touches patient data (hosting, any
  model provider).
- Incident response and breach notification procedures; contingency plan.
- For SOC 2: a scoped system description, the controls above operated and evidenced over time (change management,
  access reviews, vendor management, monitoring, vulnerability management), and an examination by a CPA firm. Type I
  is a point in time; Type II covers a period, usually 3 to 12 months.

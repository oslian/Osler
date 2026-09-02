"""Medication Safety blueprint.

Import a case (free text or form) -> the agent parses it and the symbolic engine
ranks candidates -> a mind-map reasoning graph -> chat to ask why.

The ranking is produced by ``engine.reasoning_engine`` and is fully
deterministic. The LLM only parses free text and explains a result that is
already fixed; it never changes the ranking, invents a dose, or makes a medical
decision. Knowledge sources are all research estimates: ``data/drugs_pkpd.json``
(drug deltas), the organ JSONs read through ``disease_world`` (a local disease
"wiki") and optionally live openFDA labels via ``agent.enrich_openfda``.

Nothing here diagnoses, prescribes or raises a clinical alert.
"""
from __future__ import annotations
import json

from flask import Blueprint, request, jsonify, Response

from ..config import MEDICATION_DATA_DIR
from . import agent, case_targets, llm_client
from .engine import reasoning_engine as RE

bp = Blueprint("medication", __name__)

DRUGS_PKPD = RE.load_knowledge("drugs_pkpd.json")["drugs"]


def _load_optional(name: str) -> dict:
    p = MEDICATION_DATA_DIR / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace")).get("drugs", {})
    except Exception:
        return {}


CLINICAL = _load_optional("demo_clinical_data.json")
SAMPLE_CASES = json.loads((MEDICATION_DATA_DIR / "sample_cases.json").read_text(encoding="utf-8"))["cases"]
# patient_id -> last analyze bundle, used to ground chat.
_CACHE: dict[str, dict] = {}


@bp.route("/api/cases")
def api_cases():
    return jsonify({"cases": SAMPLE_CASES, "indications": case_targets.list_indications(),
                    "env_llm": llm_client.available()})


@bp.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json(force=True) or {}
    out = agent.analyze(d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                        text=d.get("text") or "", api_key=(d.get("api_key") or "").strip() or None,
                        provider=d.get("provider") or None, use_openfda=bool(d.get("use_openfda")))
    if "error" in out:
        return jsonify(out), 400
    pid = d.get("patient_id") or out["indication"]
    out["patient_id"] = pid
    _CACHE[pid] = out
    return jsonify(out)


@bp.route("/api/analyze_stream", methods=["POST"])
def api_analyze_stream():
    """NDJSON stream: one JSON object per line, one per workflow step, ending with
    {type:final, bundle}. Lets the UI render the agent's steps live (Claude-Code style)."""
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id") or "p"
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # deterministic fixed pipeline, or LLM tool-calling orchestration
    if d.get("orchestrator") == "llm":
        from . import orchestrator
        stream_fn = orchestrator.analyze_stream_llm
    else:
        stream_fn = agent.analyze_stream

    def gen():
        trace = []
        try:
            for ev in stream_fn(
                    d.get("fields") or {}, DRUGS_PKPD, dict(CLINICAL),
                    text=d.get("text") or "", api_key=api_key,
                    provider=provider, use_openfda=bool(d.get("use_openfda"))):
                if ev["type"] == "step" and not ev.get("running"):
                    trace.append({"icon": ev["icon"], "title": ev["title"], "detail": ev.get("detail", "")})
                if ev["type"] == "final":
                    ev["bundle"]["trace"] = trace
                    ev["bundle"]["patient_id"] = pid
                    _CACHE[pid] = ev["bundle"]
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # never break the stream silently
            yield json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}) + "\n"

    return Response(gen(), mimetype="application/x-ndjson")


_SYSTEM = (
    "You explain a research-only decision-support candidate ranking. "
    "Your ONLY job is to explain, in clear language, the candidate ranking that the "
    "symbolic reasoning engine has ALREADY computed (provided below).\n"
    "Rules:\n"
    "1) Do not make independent medical decisions, change the ranking, or suggest drugs not "
    "in the candidate list.\n"
    "2) Explain only from the provided data: disease perturbations (the local world model), "
    "mechanism_chain, matched_targets, clinical_role, safety, final_answer.\n"
    "3) Never invent a dose; only cite dose.verbatim when present, and note a clinician must "
    "confirm it.\n"
    "4) If asked about a drug not in the list, say it was not matched and why "
    "(mechanism mismatch / not in the drug knowledge base).\n"
    "5) delta magnitudes and directions are illustrative. Decision support only — "
    "a licensed clinician makes the final call. This system does not diagnose or prescribe.\n"
    "Answer concisely, in the user's language, citing specific drugs and target variables."
)


def _ground(bundle: dict) -> str:
    res = bundle["result"]
    dm = bundle.get("disease_model", {})
    p = res["patient"]
    lines = [f"Indication: {res.get('indication_label', res.get('indication'))}",
             f"Treatment targets: {', '.join(res.get('target_states', []))}",
             f"Disease world-model ({dm.get('source')}): {dm.get('disease')}",
             "  perturbations: " + ", ".join(
                 f"{pp['variable']} {'↑' if pp['direction'] == 'high' else '↓'}"
                 for pp in dm.get("perturbations", [])),
             "  symptoms: " + ", ".join(dm.get("symptoms", [])),
             f"Patient: age={p.get('age')}, renal={p.get('renal_label')}, "
             f"allergies={p.get('allergies')}, meds={p.get('meds')}, flags={p.get('flags')}",
             f"Mode: {'mechanism-only' if res.get('mechanism_only') else 'mechanism + label safety gate'}",
             "", "Ranked candidates:"]
    for i, c in enumerate(res.get("candidates", []), 1):
        dose = c.get("dose", {})
        dose_txt = dose.get("verbatim") if dose.get("patient_specific_allowed") else None
        reasons = "; ".join(r.get("message", "") for r in c.get("safety", {}).get("reasons", []))
        lines.append(
            f"{i}. {c['drug']} [{c['clinical_role'].get('label')}] score={c.get('mechanism_score')}\n"
            f"   mechanism: {c.get('mechanism_chain')}\n"
            f"   matched: {[m['target'] + '/' + m['effect_type'] for m in c.get('matched_targets', [])]}\n"
            f"   safety: {c['safety'].get('decision')} ({reasons})\n"
            f"   dose: {dose_txt or 'not shown'}\n"
            f"   conclusion: {c.get('final_answer')}")
    return "\n".join(lines)


@bp.route("/api/chat", methods=["POST"])
def api_chat():
    d = request.get_json(force=True) or {}
    pid = d.get("patient_id")
    messages = d.get("messages") or []
    api_key = (d.get("api_key") or "").strip() or None
    provider = d.get("provider") or None
    # Prefer grounding sent by the client (survives server restarts); fall back to cache.
    bundle = None
    if d.get("result"):
        bundle = {"result": d["result"], "disease_model": d.get("disease_model") or {}}
    elif pid in _CACHE:
        bundle = _CACHE[pid]
    if not bundle:
        return jsonify({"reply": "Please analyze a case first so I can explain its result.",
                        "ok": False})
    system = _SYSTEM + "\n\n=== CURRENT CASE RESULT ===\n" + _ground(bundle)
    ok, text = llm_client.chat(system, messages, api_key=api_key, provider=provider)
    return jsonify({"reply": text, "ok": ok})

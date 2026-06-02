"""
llm.py — the generative "smart brain" (provider-agnostic).

It lets the bot answer freely — including questions we never pre-wrote — in the
customer's own language, while staying grounded in Cargoteer's real facts.

Provider is auto-selected from whichever key is set:
  • Gemini  (GEMINI_API_KEY)  — free tier, strong Albanian + English
  • OpenAI  (OPENAI_API_KEY)  — pay-as-you-go fallback
If none is set (or a call fails for any reason) generate_reply() returns None,
so the caller falls back to the free local brain (brain.py). The bot is never
down.

Grounding: the company's actual facts are pulled from knowledge.json and added
to the system prompt, so the LLM uses our real hours/prices/policies instead of
inventing them — single source of truth, no duplication.
"""

import requests

import config

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Built once from knowledge.json the first time it's needed.
_system_prompt_cache = None
# Reused OpenAI client (created lazily so a missing key never breaks startup).
_openai_client = None


def active_provider():
    """Return 'gemini', 'openai', or None — which LLM (if any) will be used."""
    forced = config.LLM_PROVIDER
    if forced == "local":
        return None
    if forced == "gemini":
        return "gemini" if config.GEMINI_API_KEY else None
    if forced == "openai":
        return "openai" if config.OPENAI_API_KEY else None
    # auto
    if config.GEMINI_API_KEY:
        return "gemini"
    if config.OPENAI_API_KEY:
        return "openai"
    return None


def is_enabled():
    return active_provider() is not None


def generate_reply(history):
    """`history` is a list of {"role": "user"|"assistant", "content": str}.
    Returns the assistant's reply text, or None if no LLM is available/working."""
    provider = active_provider()
    if provider is None:
        return None
    try:
        if provider == "gemini":
            return _gemini_reply(history)
        return _openai_reply(history)
    except Exception as e:  # never crash the webhook — let the local brain take over
        print(f"[llm] {provider} call failed, falling back to local brain: {e}")
        return None


# --- grounding ----------------------------------------------------------------

def _system_prompt():
    """Behavioural rules (config.SYSTEM_PROMPT) + company facts from the
    knowledge base, so the LLM answers with our real info."""
    global _system_prompt_cache
    if _system_prompt_cache is None:
        from brain import brain  # imported lazily to avoid a circular import
        facts = _company_facts(brain.knowledge)
        _system_prompt_cache = (
            f"{config.SYSTEM_PROMPT}\n\n"
            f"COMPANY FACTS (the only source for specifics):\n{facts}"
        )
    return _system_prompt_cache


def refresh_grounding():
    """Drop the cached system prompt so the next reply re-reads knowledge.json
    (call after brain.learn() or editing the knowledge base at runtime)."""
    global _system_prompt_cache
    _system_prompt_cache = None


def _company_facts(knowledge):
    """One factual line per intent, taken from its English response (the same
    text the local brain uses), so both brains stay perfectly consistent."""
    lines = []
    for intent in knowledge.get("intents", []):
        resp = intent.get("responses", {})
        en = resp.get("en") if isinstance(resp, dict) else resp
        if en:
            fact = " ".join(en[0].split())  # collapse newlines/extra spaces
            lines.append(f"- {intent['tag']}: {fact}")
    return "\n".join(lines)


# --- providers ----------------------------------------------------------------

def _gemini_reply(history):
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in history
    ]
    payload = {
        "system_instruction": {"parts": [{"text": _system_prompt()}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 400},
    }
    url = _GEMINI_URL.format(model=config.GEMINI_MODEL)
    resp = requests.post(
        url, params={"key": config.GEMINI_API_KEY}, json=payload, timeout=20
    )
    if resp.status_code != 200:
        print(f"[llm] gemini HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        # e.g. blocked by a safety filter — let the local brain answer.
        print(f"[llm] gemini returned no candidates: {str(data)[:200]}")
        return None
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or None


def _openai_reply(history):
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=15)
    messages = [{"role": "system", "content": _system_prompt()}]
    messages.extend(history)
    resp = _openai_client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip() or None

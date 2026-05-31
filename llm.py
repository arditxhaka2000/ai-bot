"""
llm.py — optional OpenAI brain.

If OPENAI_API_KEY is set, generate_reply() asks gpt-4o-mini for a reply in the
user's own language. If the key is missing or the call fails for any reason,
it returns None so the caller can fall back to the free local brain.
"""

import config

# The client is created lazily so a missing key never breaks startup.
_client = None


def is_enabled():
    return bool(config.OPENAI_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=15)
    return _client


def generate_reply(history):
    """`history` is a list of {"role": "user"|"assistant", "content": str}.
    Returns the assistant's reply text, or None if OpenAI is unavailable."""
    if not is_enabled():
        return None

    try:
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.extend(history)

        resp = _get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        # Never crash the webhook — log and let the local brain take over.
        print(f"[llm] OpenAI call failed, falling back to local brain: {e}")
        return None

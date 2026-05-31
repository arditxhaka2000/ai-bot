"""
responder.py — decides how to answer each user.

Strategy: try the OpenAI brain first (smart, multilingual). If it's disabled
or fails, fall back to the free local brain. Keeps a short per-user history so
the OpenAI brain can hold a real conversation.
"""

from collections import defaultdict, deque

import config
import llm
from brain import brain

# sender_id -> recent turns, capped at HISTORY_TURNS*2 messages (user+assistant).
_history = defaultdict(lambda: deque(maxlen=config.HISTORY_TURNS * 2))


def respond(sender_id, text):
    """Return the reply for one user message and update that user's history."""
    history = _history[sender_id]
    history.append({"role": "user", "content": text})

    reply = llm.generate_reply(list(history))
    source = "openai"
    if reply is None:
        # OpenAI disabled or failed — use the local brain (no memory needed).
        reply = brain.reply(text)
        source = "local"

    history.append({"role": "assistant", "content": reply})
    print(f"[responder] ({source}) {text!r} -> {reply!r}")
    return reply


def reset(sender_id):
    """Forget a user's conversation history."""
    _history.pop(sender_id, None)

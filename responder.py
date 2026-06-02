"""
responder.py — decides how to answer each user.

Strategy: try the OpenAI brain first (smart, multilingual). If it's disabled
or fails, fall back to the free local brain. Keeps a short per-user history so
the OpenAI brain can hold a real conversation, plus a little local-brain state
(e.g. "I just asked this user for a tracking number") so the fallback can run
multi-turn flows too.
"""

from collections import defaultdict, deque

import config
import llm
from brain import brain

# sender_id -> recent turns, capped at HISTORY_TURNS*2 messages (user+assistant).
_history = defaultdict(lambda: deque(maxlen=config.HISTORY_TURNS * 2))
# sender_id -> what the local brain is waiting for next (e.g. "tracking").
_local_state = defaultdict(dict)


def respond(sender_id, text):
    """Return the reply for one user message and update that user's history."""
    history = _history[sender_id]
    history.append({"role": "user", "content": text})

    reply = llm.generate_reply(list(history))
    source = llm.active_provider() or "?"
    if reply is None:
        # OpenAI disabled or failed — use the local brain, with its own
        # per-user follow-up state so multi-turn flows work offline too.
        result = brain.reply_with_state(text, context=_local_state[sender_id])
        reply = result["reply"]
        # Remember what's expected next AND the language in use, so a follow-up
        # with no language signal (e.g. a bare tracking number) stays in it.
        _local_state[sender_id] = {
            "awaiting": result["awaiting"],
            "lang": result["lang"],
        }
        source = f"local:{result['lang']}:{result['intent']}"
    else:
        # The smart brain handled it; don't leave a stale "awaiting" hanging.
        _local_state[sender_id] = {}

    history.append({"role": "assistant", "content": reply})
    print(f"[responder] ({source}) {text!r} -> {reply!r}")
    return reply


def reset(sender_id):
    """Forget a user's conversation history and local-brain state."""
    _history.pop(sender_id, None)
    _local_state.pop(sender_id, None)

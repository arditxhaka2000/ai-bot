"""
responder.py — decides how to answer each user, and orchestrates everything.

Order of handling for one message:
  1) Rate limit       — shield against spam / runaway LLM cost.
  2) Human handoff     — if the user asked for a person, go quiet and let the
                         team take over (until they ask for the bot again).
  3) Tracking number   — look the shipment up and answer with the REAL status
                         (accurate, instant, free — no LLM guessing).
  4) Instant quote     — if weight + destination are both given, compute a real
                         price from the rate card.
  5) LLM brain         — generative, on-brand, multilingual (Gemini/OpenAI).
  6) Local brain       — always-on fallback if the LLM is unavailable.

Everything is logged to conversations.log (JSONL) for audit.
"""

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import config
import llm
import quotes
import tracking
from brain import brain, detect_language, extract_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
CONVO_LOG = os.path.join(HERE, "conversations.log")

# --- per-user state -----------------------------------------------------------
_history = defaultdict(lambda: deque(maxlen=config.HISTORY_TURNS * 2))
_local_state = defaultdict(dict)          # sender -> {awaiting, lang}
_handoff = {}                             # sender -> True while a human is in
_rate = defaultdict(lambda: deque(maxlen=64))  # sender -> recent timestamps

# --- rate limiting ------------------------------------------------------------
RATE_MAX = 20          # messages...
RATE_WINDOW = 60       # ...per this many seconds

# --- handoff / resume triggers (folded text, substring match) -----------------
_HANDOFF_TRIGGERS = (
    "talk to a human", "talk to a person", "speak to someone", "real person",
    "live agent", "customer service", "representative", "agent", "operator",
    "dua nje operator", "dua te flas me nje njeri", "me lidh me operator",
    "flas me dike", "perfaqesues", "njeri real",
)
_RESUME_TRIGGERS = ("bot", "boti", "assistant", "asistent", "menu", "menu kryesore",
                    "start over", "rifillo")


def _t(lang, en, sq):
    return sq if lang == "sq" else en


def _quick_replies(lang):
    if lang == "sq":
        return ["📦 Gjurmo", "💰 Çmimi", "🚚 Koha", "🧑 Operator"]
    return ["📦 Track", "💰 Pricing", "🚚 Delivery time", "🧑 Agent"]


def _lang_for(sender, text):
    """Detected language, sticky to the conversation when the message has none."""
    return detect_language(text, fallback=_local_state[sender].get("lang", "en"))


def respond(sender_id, text, channel="messenger"):
    """Return {reply, quick_replies, source, handoff}. `reply` is None when the
    bot should stay silent (e.g. a human has taken over)."""
    text = (text or "").strip()
    lang = _lang_for(sender_id, text)

    # 1) Rate limit ------------------------------------------------------------
    if _rate_limited(sender_id):
        msg = _t(lang,
                 "You're sending messages very fast — give me a sec to keep up. 🙏",
                 "Po dërgoni shumë shpejt — më jepni një moment. 🙏")
        return _finish(sender_id, text, msg, None, "ratelimit", channel, lang)

    # 2) Human handoff ---------------------------------------------------------
    folded = text.lower()
    if _handoff.get(sender_id):
        if any(k == folded or k in folded.split() for k in _RESUME_TRIGGERS):
            _handoff.pop(sender_id, None)
            msg = _t(lang, "I'm back! 🤖 How can I help?",
                     "U ktheva! 🤖 Si mund t'ju ndihmoj?")
            return _finish(sender_id, text, msg, _quick_replies(lang),
                           "resume", channel, lang)
        # A person is handling this chat — stay silent.
        return _finish(sender_id, text, None, None, "handoff-silent", channel, lang)

    if _wants_human(folded):
        _handoff[sender_id] = True
        msg = _t(lang,
                 "Sure — I'm connecting you with a Cargoteer team member. "
                 "They'll reply right here. You can also reach us at "
                 "support@cargoteer.com / +355 00 000 000. (Type 'bot' anytime "
                 "to use the assistant again.)",
                 "Sigurisht — po ju lidh me një anëtar të ekipit Cargoteer. "
                 "Do t'ju përgjigjen këtu. Mund të na shkruani edhe në "
                 "support@cargoteer.com / +355 00 000 000. (Shkruani 'bot' "
                 "kurdo për të përdorur sërish asistentin.)")
        return _finish(sender_id, text, msg, None, "handoff", channel, lang)

    # 3) Tracking number -> real status ---------------------------------------
    number = extract_tracking(text)
    awaiting = _local_state[sender_id].get("awaiting")
    if number and (awaiting == "tracking" or _mostly_tracking(text, number)):
        reply = _tracking_reply(number, lang)
        qr = (["📦 " + _t(lang, "Track another", "Gjurmo tjetër"),
               "🧑 " + _t(lang, "Agent", "Operator")])
        _local_state[sender_id]["awaiting"] = None
        return _finish(sender_id, text, reply, qr, "tracking", channel, lang)

    # 4) Instant quote when weight + destination are both present --------------
    weight = quotes.parse_weight(text)
    zone = quotes.guess_zone(text)
    if weight and zone:
        q = quotes.estimate(weight, zone)
        if q:
            reply = quotes.format_estimate(q, lang)
            return _finish(sender_id, text, reply, _quick_replies(lang),
                           "quote", channel, lang)

    # 5) LLM brain -------------------------------------------------------------
    history = _history[sender_id]
    history.append({"role": "user", "content": text})
    reply = llm.generate_reply(list(history))
    source = llm.active_provider() or "?"

    # 6) Local brain fallback --------------------------------------------------
    if reply is None:
        result = brain.reply_with_state(text, context=_local_state[sender_id])
        reply = result["reply"]
        _local_state[sender_id] = {"awaiting": result["awaiting"],
                                   "lang": result["lang"]}
        lang = result["lang"]
        source = f"local:{result['intent']}"
    else:
        _local_state[sender_id]["lang"] = lang

    history.append({"role": "assistant", "content": reply})
    qr = _quick_replies(lang) if source.startswith("local") else None
    return _finish(sender_id, text, reply, qr, source, channel, lang)


# --- helpers ------------------------------------------------------------------

def _rate_limited(sender_id):
    now = time.monotonic()
    bucket = _rate[sender_id]
    while bucket and now - bucket[0] > RATE_WINDOW:
        bucket.popleft()
    bucket.append(now)
    return len(bucket) > RATE_MAX


def _wants_human(folded):
    return any(k == folded or k in folded for k in _HANDOFF_TRIGGERS)


def _mostly_tracking(text, number):
    leftover = text.lower().replace(number.lower(), " ").split()
    return len(leftover) <= 3


def _tracking_reply(number, lang):
    record = tracking.lookup(number)
    if record:
        return tracking.format_status(record, lang)
    return _t(
        lang,
        f"I couldn't find shipment {number.upper()} yet. If it was just "
        f"created, status can take a few hours to appear. Want me to forward "
        f"it to our team to check? (type 'agent')",
        f"Nuk e gjeta dërgesën {number.upper()} ende. Nëse sapo është krijuar, "
        f"statusi mund të shfaqet pas pak orësh. Doni t'ua përcjell ekipit për "
        f"kontroll? (shkruani 'operator')",
    )


def _finish(sender_id, text_in, reply, quick_replies, source, channel, lang):
    _log(sender_id, text_in, reply, source, channel, lang)
    return {"reply": reply, "quick_replies": quick_replies,
            "source": source, "handoff": bool(_handoff.get(sender_id))}


def _log(sender_id, text_in, reply, source, channel, lang):
    print(f"[responder] ({source}) {text_in!r} -> {reply!r}")
    try:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel": channel,
            "sender": sender_id,
            "lang": lang,
            "source": source,
            "in": text_in,
            "out": reply,
        }, ensure_ascii=False)
        with open(CONVO_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # never let logging break a reply


def reset(sender_id):
    """Forget a user's conversation history and state."""
    _history.pop(sender_id, None)
    _local_state.pop(sender_id, None)
    _handoff.pop(sender_id, None)
    _rate.pop(sender_id, None)

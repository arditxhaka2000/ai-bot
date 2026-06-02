"""
responder.py — decides how to answer each user, and orchestrates everything.

Order of handling for one message:
  0) Get Started      — welcome a brand-new contact.
  1) Rate limit       — shield against spam / runaway LLM cost.
  2) Human handoff     — if the user asked for a person (or escalated), go quiet
                         and let the team take over (until they ask for the bot).
  3) Auto-escalation   — repeated frustration hands off to a human + notifies.
  4) Tracking number   — look the shipment up and answer with the REAL status.
  5) Instant quote     — weight + destination -> a real price from the rate card.
  6) LLM brain         — generative, on-brand, multilingual (Gemini/OpenAI).
  7) Local brain       — always-on fallback if the LLM is unavailable.

State (history, language, handoff) persists in SQLite (store.py) so the bot
remembers across restarts. Actionable requests are captured as leads for the
team, and the team can be pinged via a webhook on handoff/new leads.
"""

import time
from collections import defaultdict, deque

import requests

import config
import llm
import quotes
import store
import tracking
from brain import brain, detect_language, extract_tracking

store.init()

# Audit log path kept for backward-compat references; audit now lives in SQLite.
CONVO_LOG = config.DB_PATH

# --- transient (in-memory) state — fine to lose on restart --------------------
_rate = defaultdict(lambda: deque(maxlen=64))   # sender -> recent timestamps
_frustration = defaultdict(int)                 # sender -> consecutive count

RATE_MAX = 20          # messages...
RATE_WINDOW = 60       # ...per this many seconds
ESCALATE_AFTER = 2     # consecutive frustrated messages -> human

# Intents worth capturing as a team to-do.
_ACTIONABLE = {"book_shipment", "schedule_pickup", "address_change",
               "cancel", "returns", "damaged_lost"}

_HANDOFF_TRIGGERS = (
    "talk to a human", "talk to a person", "speak to someone", "real person",
    "live agent", "customer service", "representative", "agent", "operator",
    "dua nje operator", "dua te flas me nje njeri", "me lidh me operator",
    "flas me dike", "perfaqesues", "njeri real",
)
_RESUME_TRIGGERS = ("bot", "boti", "assistant", "asistent", "menu",
                    "start over", "rifillo")


def _t(lang, en, sq):
    return sq if lang == "sq" else en


def _quick_replies(lang):
    if lang == "sq":
        return ["📦 Gjurmo", "💰 Çmimi", "🚚 Koha", "🧑 Operator"]
    return ["📦 Track", "💰 Pricing", "🚚 Delivery time", "🧑 Agent"]


def respond(sender_id, text, channel="messenger"):
    """Return {reply, quick_replies, source, handoff}. `reply` is None when the
    bot should stay silent (e.g. a human has taken over)."""
    text = (text or "").strip()
    state = store.get_state(sender_id)
    lang = detect_language(text, fallback=state.get("lang") or "en")

    # 0) Get Started ----------------------------------------------------------
    if text == "GET_STARTED":
        msg = ("Welcome to Cargoteer! 👋 I can track your shipment, give "
               "delivery times & rates, book a pickup, or connect you to a "
               "human — in English or Albanian. How can I help?")
        return _finish(sender_id, "GET_STARTED", msg, _quick_replies(lang),
                       "get_started", channel, lang)

    if not text:
        return _finish(sender_id, text, None, None, "empty", channel, lang)

    store.add_message(sender_id, "user", text, channel)

    # 1) Rate limit -----------------------------------------------------------
    if _rate_limited(sender_id):
        msg = _t(lang,
                 "You're sending messages very fast — give me a sec to keep up. 🙏",
                 "Po dërgoni shumë shpejt — më jepni një moment. 🙏")
        return _finish(sender_id, text, msg, None, "ratelimit", channel, lang)

    folded = text.lower()

    # 2) Human handoff (already in handoff, or requested) ---------------------
    if state.get("handoff"):
        if any(k == folded or k in folded.split() for k in _RESUME_TRIGGERS):
            store.set_state(sender_id, handoff=False)
            msg = _t(lang, "I'm back! 🤖 How can I help?",
                     "U ktheva! 🤖 Si mund t'ju ndihmoj?")
            return _finish(sender_id, text, msg, _quick_replies(lang),
                           "resume", channel, lang)
        return _finish(sender_id, text, None, None, "handoff-silent",
                       channel, lang)

    if _wants_human(folded):
        return _do_handoff(sender_id, text, channel, lang,
                           kind="handoff_request", source="handoff")

    # 3) Auto-escalation on repeated frustration ------------------------------
    if brain.is_frustrated(text):
        _frustration[sender_id] += 1
        if _frustration[sender_id] >= ESCALATE_AFTER:
            return _do_handoff(sender_id, text, channel, lang,
                               kind="complaint", source="escalation")
    else:
        _frustration[sender_id] = 0

    # Capture actionable requests for the team (works on any answer path).
    _maybe_capture_lead(sender_id, text, channel)

    # 4) Tracking number -> real status ---------------------------------------
    number = extract_tracking(text)
    if number and (state.get("awaiting") == "tracking"
                   or _mostly_tracking(text, number)):
        reply = _tracking_reply(number, lang)
        qr = ["📦 " + _t(lang, "Track another", "Gjurmo tjetër"),
              "🧑 " + _t(lang, "Agent", "Operator")]
        store.set_state(sender_id, awaiting=None, lang=lang)
        return _finish(sender_id, text, reply, qr, "tracking", channel, lang)

    # 5) Instant quote when weight + destination are both present -------------
    weight = quotes.parse_weight(text)
    zone = quotes.guess_zone(text)
    if weight and zone:
        q = quotes.estimate(weight, zone)
        if q:
            reply = quotes.format_estimate(q, lang)
            store.set_state(sender_id, lang=lang)
            return _finish(sender_id, text, reply, _quick_replies(lang),
                           "quote", channel, lang)

    # 6) LLM brain ------------------------------------------------------------
    history = store.get_history(sender_id, config.HISTORY_TURNS * 2)
    reply = llm.generate_reply(history)
    source = llm.active_provider() or "?"

    # 7) Local brain fallback -------------------------------------------------
    if reply is None:
        result = brain.reply_with_state(
            text, context={"awaiting": state.get("awaiting"),
                           "lang": state.get("lang")})
        reply = result["reply"]
        lang = result["lang"]
        store.set_state(sender_id, awaiting=result["awaiting"], lang=lang)
        source = f"local:{result['intent']}"
        qr = _quick_replies(lang)
    else:
        store.set_state(sender_id, lang=lang)
        qr = None

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


def _do_handoff(sender_id, text, channel, lang, kind, source):
    store.set_state(sender_id, handoff=True, lang=lang)
    store.add_lead(sender_id, channel, kind, text)
    _notify(f"🔔 Handoff ({kind}) for {sender_id} [{channel}]: {text!r}")
    if source == "escalation":
        msg = _t(lang,
                 "I'm sorry this has been frustrating. 🙏 I'm bringing in a "
                 "Cargoteer team member to help you directly — they'll reply "
                 "here shortly. (type 'bot' anytime to use the assistant again.)",
                 "Më vjen keq që ka qenë e bezdisshme. 🙏 Po sjell një anëtar të "
                 "ekipit Cargoteer që t'ju ndihmojë drejtpërdrejt — do t'ju "
                 "përgjigjen këtu shpejt. (shkruani 'bot' për asistentin.)")
    else:
        msg = _t(lang,
                 "Sure — I'm connecting you with a Cargoteer team member. "
                 "They'll reply right here. You can also reach us at "
                 "support@cargoteer.com / +355 00 000 000. (Type 'bot' anytime "
                 "to use the assistant again.)",
                 "Sigurisht — po ju lidh me një anëtar të ekipit Cargoteer. "
                 "Do t'ju përgjigjen këtu. Mund të na shkruani edhe në "
                 "support@cargoteer.com / +355 00 000 000. (Shkruani 'bot' "
                 "kurdo për asistentin.)")
    return _finish(sender_id, text, msg, None, source, channel, lang)


def _maybe_capture_lead(sender_id, text, channel):
    tag, score = brain.classify(text)
    if tag in _ACTIONABLE and score >= 0.5:
        if store.add_lead(sender_id, channel, tag, text):
            _notify(f"📝 New lead ({tag}) from {sender_id} [{channel}]: {text!r}")


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


def _notify(message):
    """Best-effort ping to the team's webhook (Slack/Discord/etc.)."""
    url = config.ADMIN_WEBHOOK_URL
    if not url:
        return
    try:
        requests.post(url, json={"text": message}, timeout=5)
    except requests.RequestException as e:
        print(f"[responder] notify failed: {e}")


def _finish(sender_id, text_in, reply, quick_replies, source, channel, lang):
    if reply:
        store.add_message(sender_id, "assistant", reply, channel)
    store.log_turn(sender_id, channel, lang, source, text_in, reply)
    print(f"[responder] ({source}) {text_in!r} -> {reply!r}")
    return {"reply": reply, "quick_replies": quick_replies,
            "source": source, "handoff": store.get_state(sender_id)["handoff"]}


def reset(sender_id):
    """Forget a user's conversation history and state."""
    store.reset_sender(sender_id)
    _rate.pop(sender_id, None)
    _frustration.pop(sender_id, None)


# Backward-compatible alias used by app.py's attachment handler.
def _log(sender_id, text_in, reply, source, channel, lang):
    store.log_turn(sender_id, channel, lang, source, text_in, reply)
    print(f"[responder] ({source}) {text_in!r} -> {reply!r}")

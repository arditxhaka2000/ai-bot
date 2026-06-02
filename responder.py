"""
responder.py — decides how to answer each carrier, and orchestrates everything.

Order of handling for one message:
  0) Get Started      — welcome a brand-new contact.
  1) Rate limit       — shield against spam / runaway LLM cost.
  2) Human handoff     — if they asked for a person (or escalated), go quiet and
                         let a teammate take over (until they ask to continue).
  3) Auto-escalation   — repeated frustration hands the chat to a teammate.
  4) LLM brain         — generative, on-brand dispatch rep (Gemini/OpenAI).
  5) Local brain       — always-on fallback if the LLM is unavailable.

State (history, language, handoff) persists in SQLite/Turso (store.py) so the
bot remembers across restarts. Actionable requests (pricing, competitor rate,
get-started) are captured as leads for the team, who can be pinged via webhook.
"""

import re
import time
from collections import defaultdict, deque

import requests

import config
import llm
import store
from brain import brain, detect_language

# Contact details a carrier might share so the team can follow up.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")

store.init()

CONVO_LOG = config.DB_PATH  # kept for backward-compat references

# --- transient (in-memory) state — fine to lose on restart --------------------
_rate = defaultdict(lambda: deque(maxlen=64))   # sender -> recent timestamps
_frustration = defaultdict(int)                 # sender -> consecutive count

RATE_MAX = 20          # messages...
RATE_WINDOW = 60       # ...per this many seconds
ESCALATE_AFTER = 2     # consecutive frustrated messages -> human

# Intents worth capturing as a team to-do (must match tags in knowledge.json).
_ACTIONABLE = {"get_started", "pricing", "competitor_pricing"}

_HANDOFF_TRIGGERS = (
    # English
    "talk to a human", "talk to a person", "speak to someone", "real person",
    "live agent", "customer service", "representative", "agent", "operator",
    "talk to a dispatcher", "speak to the team", "call me",
    # Spanish
    "hablar con una persona", "hablar con alguien", "persona real", "agente",
    "operador", "representante", "servicio al cliente", "llamenme",
    "hablar con un despachador",
    # Russian
    "оператор", "человек", "менеджер", "живой человек", "поговорить с человеком",
    "свяжите с человеком", "перезвоните мне", "позвоните мне",
)
_RESUME_TRIGGERS = ("bot", "assistant", "menu", "start over", "continue",
                    "continuar", "seguir", "продолжить", "бот", "меню")


def _loc(lang, **by_lang):
    """Pick a localized string by language, falling back to English."""
    return by_lang.get(lang) or by_lang["en"]


def _quick_replies(lang):
    if lang == "es":
        return ["💲 Precio", "🚚 Cómo funciona", "📝 Empezar", "🧑 Hablar con el equipo"]
    if lang == "ru":
        return ["💲 Цена", "🚚 Как работает", "📝 Начать", "🧑 Связаться с командой"]
    return ["💲 Pricing", "🚚 How it works", "📝 Get started", "🧑 Talk to a rep"]


def respond(sender_id, text, channel="messenger"):
    """Return {reply, quick_replies, source, handoff}. `reply` is None when the
    bot should stay silent (e.g. a teammate has taken over)."""
    text = (text or "").strip()
    state = store.get_state(sender_id)
    lang = detect_language(text, fallback=state.get("lang") or "en")

    # 0) Get Started ----------------------------------------------------------
    if text == "GET_STARTED":
        msg = _loc(
            lang,
            en="Welcome to Cargoteer! 👋 We dispatch for owner-operators and "
               "small fleets — finding the best-paying loads, negotiating "
               "rates, and handling brokers and paperwork. What are you "
               "running, and how can we help?",
            es="¡Bienvenido a Cargoteer! 👋 Despachamos para operadores-"
               "propietarios y flotas pequeñas — buscamos las cargas mejor "
               "pagadas, negociamos tarifas y manejamos brokers y papeleo. "
               "¿Qué manejas y cómo te ayudamos?",
            ru="Добро пожаловать в Cargoteer! 👋 Мы диспетчеры для водителей-"
               "собственников и небольших автопарков — находим самые выгодные "
               "грузы, договариваемся о ставках и берём на себя брокеров и "
               "документы. На чём работаете и чем помочь?")
        return _finish(sender_id, "GET_STARTED", msg, _quick_replies(lang),
                       "get_started", channel, lang)

    if not text:
        return _finish(sender_id, text, None, None, "empty", channel, lang)

    store.add_message(sender_id, "user", text, channel)

    # 1) Rate limit -----------------------------------------------------------
    if _rate_limited(sender_id):
        msg = _loc(
            lang,
            en="Whoa, lots of messages coming in — give me a sec to catch up. 🙏",
            es="¡Cuántos mensajes! Dame un segundo para ponerme al día. 🙏",
            ru="Ого, много сообщений — дайте секунду, чтобы успеть. 🙏")
        return _finish(sender_id, text, msg, None, "ratelimit", channel, lang)

    # Always capture shared contact info, whatever the message routes to.
    _maybe_capture_contact(sender_id, text, channel)

    folded = text.lower()

    # 2) Human handoff (already in handoff, or requested) ---------------------
    if state.get("handoff"):
        if any(k == folded or k in folded.split() for k in _RESUME_TRIGGERS):
            store.set_state(sender_id, handoff=False)
            msg = _loc(
                lang,
                en="I'm here and happy to keep helping — what do you need?",
                es="Aquí estoy y con gusto sigo ayudándote — ¿qué necesitas?",
                ru="Я на связи и рад продолжить помогать — что вам нужно?")
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

    # 4) LLM brain ------------------------------------------------------------
    history = store.get_history(sender_id, config.HISTORY_TURNS * 2)
    reply = llm.generate_reply(history)
    source = llm.active_provider() or "?"

    # 5) Local brain fallback -------------------------------------------------
    if reply is None:
        result = brain.reply_with_state(text, context={"lang": state.get("lang")})
        reply = result["reply"]
        lang = result["lang"]
        store.set_state(sender_id, lang=lang)
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
        msg = _loc(
            lang,
            en="I hear you, and I'm sorry for the hassle. 🙏 I'm bringing in a "
               "teammate to help you directly — they'll jump in here shortly.",
            es="Te entiendo y lamento la molestia. 🙏 Voy a traer a un compañero "
               "del equipo para ayudarte directamente — se unirá aquí en breve.",
            ru="Понимаю вас, извините за неудобства. 🙏 Подключаю коллегу, чтобы "
               "помочь напрямую — он скоро ответит здесь.")
    else:
        msg = _loc(
            lang,
            en="Absolutely — I'll get a teammate to reach out to you directly. "
               "You can also email us at dispatch@cargoteer.com. Mind sharing "
               "your equipment and lanes so we're ready to help?",
            es="¡Claro! Haré que un compañero del equipo te contacte "
               "directamente. También puedes escribirnos a dispatch@cargoteer.com. "
               "¿Me dices tu equipo y tus rutas para estar listos para ayudarte?",
            ru="Конечно — попрошу коллегу связаться с вами напрямую. Также можно "
               "написать на dispatch@cargoteer.com. Подскажите ваш прицеп и "
               "маршруты, чтобы мы были готовы помочь?")
    return _finish(sender_id, text, msg, None, source, channel, lang)


def _maybe_capture_lead(sender_id, text, channel):
    tag, score = brain.classify(text)
    if tag in _ACTIONABLE and score >= 0.5:
        if store.add_lead(sender_id, channel, tag, text):
            _notify(f"📝 New lead ({tag}) from {sender_id} [{channel}]: {text!r}")


def _maybe_capture_contact(sender_id, text, channel):
    """If the carrier shares an email/phone, save it as a high-priority lead
    WITH recent context, so the promised follow-up actually reaches the team."""
    email = _EMAIL_RE.search(text)
    phone = None if email else _PHONE_RE.search(text)
    contact = (email or phone).group(0).strip() if (email or phone) else None
    if not contact:
        return
    # Build a short transcript so the team has context (rate, equipment, lanes).
    history = store.get_history(sender_id, 12)
    convo = " | ".join(f"{m['role']}: {m['content']}" for m in history[-8:])
    lead_text = f"CONTACT: {contact} — context: {convo}"
    if store.add_lead(sender_id, channel, "contact", lead_text):
        _notify(f"📞 Carrier shared contact {contact} ({sender_id}) — follow up "
                f"with a custom quote. Context: {convo}")


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


# Backward-compatible helper used by app.py's attachment handler.
def _log(sender_id, text_in, reply, source, channel, lang):
    store.log_turn(sender_id, channel, lang, source, text_in, reply)
    print(f"[responder] ({source}) {text_in!r} -> {reply!r}")

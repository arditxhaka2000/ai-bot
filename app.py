"""
app.py — the Facebook/Instagram Messenger webhook server.

Meta sends every incoming message here as an HTTP POST. We read the event,
ask the responder (LLM first, local brain as fallback) for a reply, and send
it back via the Send API — with a typing indicator, duplicate protection, and
graceful handling of attachments, postbacks, and quick replies.

  GET  /webhook  -> Meta's one-time verification handshake
  POST /webhook  -> incoming events
  GET  /         -> health check
  GET  /stats    -> simple conversation stats (for monitoring)
"""

import threading

from flask import Flask, jsonify, request

import config
import messenger
import responder
import store

# One lock per sender so a person's messages are processed AND answered in the
# order they arrived — fixes the race where a quick 2nd message gets a reply
# before a slower 1st one. (Single gunicorn worker keeps this lock global.)
_sender_locks = {}
_locks_guard = threading.Lock()


def _lock_for(sender_id):
    with _locks_guard:
        lock = _sender_locks.get(sender_id)
        if lock is None:
            lock = threading.Lock()
            _sender_locks[sender_id] = lock
        return lock

app = Flask(__name__)


@app.get("/")
def health():
    return "Bot is running. ✅", 200


@app.get("/stats")
def stats():
    """Counters from the audit store (volume by brain, open leads, handoffs)."""
    return jsonify(store.stats())


def _admin_ok():
    """Admin endpoints require ?token= matching ADMIN_TOKEN (and it must be set)."""
    token = request.args.get("token") or request.headers.get("X-Admin-Token")
    return bool(config.ADMIN_TOKEN) and token == config.ADMIN_TOKEN


@app.get("/admin/leads")
def admin_leads():
    if not _admin_ok():
        return "Forbidden", 403
    status = request.args.get("status")  # e.g. ?status=new
    return jsonify(store.recent_leads(limit=100, status=status))


@app.get("/admin/conversations")
def admin_conversations():
    if not _admin_ok():
        return "Forbidden", 403
    return jsonify(store.recent_turns(limit=100))


@app.get("/admin/handoffs")
def admin_handoffs():
    if not _admin_ok():
        return "Forbidden", 403
    return jsonify(store.active_handoffs())


@app.get("/webhook")
def verify():
    """Meta calls this once when you set up the webhook."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == config.VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.post("/webhook")
def incoming():
    if not messenger.verify_signature(request):
        return "Bad signature", 403

    data = request.get_json(silent=True) or {}
    obj = data.get("object")
    # Both Messenger ("page") and Instagram ("instagram") use this same shape.
    if obj not in ("page", "instagram"):
        return "ok", 200
    channel = "instagram" if obj == "instagram" else "messenger"

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            try:
                _handle_event(event, channel)
            except Exception as e:  # never let one bad event 500 the webhook
                print(f"[app] error handling event: {e}")

    return "ok", 200


def _handle_event(event, channel):
    sender_id = event.get("sender", {}).get("id")
    if not sender_id:
        return
    # Serialize everything for this sender so replies go out in arrival order.
    with _lock_for(sender_id):
        _process_event(sender_id, event, channel)


def _process_event(sender_id, event, channel):
    message = event.get("message", {})
    postback = event.get("postback", {})

    # Ignore echoes of our own messages and delivery/read receipts.
    if message.get("is_echo") or "delivery" in event or "read" in event:
        return

    # Don't reply twice to a redelivered webhook.
    mid = message.get("mid") or postback.get("mid")
    if messenger.is_duplicate(mid):
        return

    # Resolve the text to act on, from the various event types.
    text = _extract_text(message, postback)

    # Attachments (photos, documents, location) with no text.
    if text is None and message.get("attachments"):
        _handle_attachment(sender_id, message["attachments"], channel)
        return

    if not text:
        return

    messenger.send_action(sender_id, "mark_seen")
    messenger.send_action(sender_id, "typing_on")
    try:
        result = responder.respond(sender_id, text, channel=channel)
    finally:
        messenger.send_action(sender_id, "typing_off")

    if result.get("reply"):
        messenger.send_message(sender_id, result["reply"],
                               quick_replies=result.get("quick_replies"))


def _extract_text(message, postback):
    """Pull the user's intent-text out of a text msg, quick reply, or postback."""
    if postback:
        # Get Started / menu buttons carry a payload (and a title).
        return postback.get("payload") or postback.get("title")
    if message.get("quick_reply"):
        return message["quick_reply"].get("payload") or message.get("text")
    return message.get("text")


def _handle_attachment(sender_id, attachments, channel):
    kinds = {a.get("type") for a in attachments}
    if "image" in kinds or "file" in kinds:
        reply = ("Thanks for sending that over! 📎 If it's your authority/MC, "
                 "insurance, or a rate con, our team will take a look. Mind "
                 "telling me your equipment and the lanes you run so we can get "
                 "you set up?")
    else:
        reply = ("Got it, thanks! Tell me what you're running and what you need "
                 "— loads, rates, or getting set up — and I'll help. 🚛")
    messenger.send_action(sender_id, "mark_seen")
    messenger.send_message(sender_id, reply)
    responder._log(sender_id, f"<attachment:{','.join(sorted(kinds))}>",
                   reply, "attachment", channel, "en")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)

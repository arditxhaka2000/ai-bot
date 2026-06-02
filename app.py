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

import os

from flask import Flask, jsonify, request

import config
import messenger
import responder

app = Flask(__name__)


@app.get("/")
def health():
    return "Bot is running. ✅", 200


@app.get("/stats")
def stats():
    """Lightweight counters from the audit log (best-effort)."""
    path = responder.CONVO_LOG
    total, by_source = 0, {}
    try:
        import json
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                total += 1
                src = (row.get("source") or "?").split(":")[0]
                by_source[src] = by_source.get(src, 0) + 1
    except FileNotFoundError:
        pass
    return jsonify({"messages_logged": total, "by_source": by_source})


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

    # Attachments (photos, etc.) with no text — e.g. a damage-claim photo.
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
    if "image" in kinds:
        reply = ("Thanks for the photo! 📷 If this is about a damaged or wrong "
                 "delivery, please also send your tracking or order number and "
                 "I'll open a claim and get our team on it. (type 'agent' for a "
                 "person)")
    elif "location" in kinds:
        reply = ("Got your location, thanks! 📍 Tell me the tracking number or "
                 "what you'd like to do and I'll help.")
    else:
        reply = ("Thanks! I can read text best — tell me in a message what you "
                 "need (tracking, pricing, a pickup…) and I'll help. 🙂")
    messenger.send_action(sender_id, "mark_seen")
    messenger.send_message(sender_id, reply)
    # Log the attachment turn too.
    responder._log(sender_id, f"<attachment:{','.join(sorted(kinds))}>",
                   reply, "attachment", channel, "en")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)

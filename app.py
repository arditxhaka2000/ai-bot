"""
app.py — the Facebook Messenger webhook server.

Facebook sends every incoming message here as an HTTP POST. We read the text,
ask the local brain for a reply, and send it back via the Send API.

  GET  /webhook  -> Facebook's one-time verification handshake
  POST /webhook  -> incoming messages
"""

from flask import Flask, request

import config
import messenger
from brain import brain

app = Flask(__name__)


@app.get("/")
def health():
    return "Bot is running. ✅", 200


@app.get("/webhook")
def verify():
    """Facebook calls this once when you set up the webhook."""
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
    if data.get("object") != "page":
        return "ok", 200

    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event.get("sender", {}).get("id")
            message = event.get("message", {})

            # Ignore echoes of our own messages and delivery receipts.
            if not sender_id or message.get("is_echo"):
                continue

            text = message.get("text")
            if not text:
                # Attachment (image, sticker, etc.) — no text to read.
                messenger.send_message(
                    sender_id,
                    "Thanks! I can only read text messages right now. 🙂",
                )
                continue

            reply = brain.reply(text)
            messenger.send_message(sender_id, reply)

    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)

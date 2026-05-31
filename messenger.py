"""Thin wrapper around the Facebook Send API + request verification."""

import hashlib
import hmac

import requests

import config

GRAPH_URL = "https://graph.facebook.com/v21.0/me/messages"


def send_message(recipient_id, text):
    """Send a text message back to a Messenger user."""
    if not config.PAGE_ACCESS_TOKEN:
        print("[messenger] PAGE_ACCESS_TOKEN not set — cannot send.")
        return

    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
        "messaging_type": "RESPONSE",
    }
    resp = requests.post(
        GRAPH_URL,
        params={"access_token": config.PAGE_ACCESS_TOKEN},
        json=payload,
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"[messenger] send failed {resp.status_code}: {resp.text}")


def verify_signature(request):
    """Confirm an incoming webhook really came from Facebook using APP_SECRET.

    Returns True if valid (or if APP_SECRET isn't configured, so you can test
    before wiring it up). Set APP_SECRET in production."""
    if not config.APP_SECRET:
        return True

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        config.APP_SECRET.encode("utf-8"),
        request.get_data(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature.split("=", 1)[1])

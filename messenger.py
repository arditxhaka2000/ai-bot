"""
messenger.py — wrapper around the Facebook/Instagram Send API + helpers.

Adds the things a real bot needs:
  • typing indicator + "seen" so the customer knows it's working
  • automatic splitting of long replies (Messenger caps text at 2000 chars)
  • a small retry on transient send failures
  • quick-reply buttons
  • request signature verification
  • duplicate-message detection (Facebook re-delivers webhooks on timeout)
"""

import hashlib
import hmac
import time
from collections import deque

import requests

import config

GRAPH_URL = "https://graph.facebook.com/v21.0/me/messages"
MAX_LEN = 1900  # a little under Messenger's 2000-char hard limit


def _post(payload, attempt_label="message"):
    """POST to the Send API with one retry on a transient error."""
    if not config.PAGE_ACCESS_TOKEN:
        print("[messenger] PAGE_ACCESS_TOKEN not set — cannot send.")
        return False
    for attempt in (1, 2):
        try:
            resp = requests.post(
                GRAPH_URL,
                params={"access_token": config.PAGE_ACCESS_TOKEN},
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            print(f"[messenger] {attempt_label} failed "
                  f"{resp.status_code}: {resp.text[:200]}")
            # 4xx (bad request/permissions) won't fix itself — don't retry.
            if 400 <= resp.status_code < 500:
                return False
        except requests.RequestException as e:
            print(f"[messenger] {attempt_label} error (attempt {attempt}): {e}")
        time.sleep(0.6)
    return False


def _split(text, limit=MAX_LEN):
    """Split a long message on paragraph/sentence boundaries where possible."""
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    chunks, current = [], ""
    for para in text.split("\n"):
        if len(current) + len(para) + 1 <= limit:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            while len(para) > limit:
                cut = para.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                chunks.append(para[:cut])
                para = para[cut:].lstrip()
            current = para
    if current:
        chunks.append(current)
    return chunks


def send_action(recipient_id, action):
    """action: 'mark_seen', 'typing_on', or 'typing_off'. Best-effort."""
    _post({"recipient": {"id": recipient_id}, "sender_action": action},
          attempt_label=f"action:{action}")


def send_message(recipient_id, text, quick_replies=None):
    """Send a text reply, split if long. quick_replies is an optional list of
    short button labels attached to the LAST chunk."""
    chunks = _split(text)
    if not chunks:
        return
    for i, chunk in enumerate(chunks):
        message = {"text": chunk}
        if quick_replies and i == len(chunks) - 1:
            message["quick_replies"] = [
                {"content_type": "text", "title": qr[:20], "payload": qr[:1000]}
                for qr in quick_replies[:11]
            ]
        _post({
            "recipient": {"id": recipient_id},
            "message": message,
            "messaging_type": "RESPONSE",
        })


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


# --- duplicate detection ------------------------------------------------------
# Facebook re-delivers a webhook if it doesn't get a fast 200. We remember the
# last N message IDs so a redelivery doesn't produce a second reply.
_seen_mids = deque(maxlen=2000)
_seen_set = set()


def is_duplicate(mid):
    """True if we've already processed this message id (and records it)."""
    if not mid:
        return False
    if mid in _seen_set:
        return True
    _seen_mids.append(mid)
    _seen_set.add(mid)
    # Keep the set bounded in step with the deque.
    while len(_seen_set) > _seen_mids.maxlen:
        _seen_set.discard(_seen_mids.popleft())
    return False

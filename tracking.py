"""
tracking.py — shipment status lookup.

This is the seam where Cargoteer's real tracking system plugs in. Today it
reads a local shipments.json (demo data) so the bot can show real-looking
statuses end to end. To go live, replace `_lookup_backend` with a call to the
client's tracking API (HTTP request, DB query, etc.) — the rest of the bot
doesn't change.

lookup(number) -> dict | None
    None means "not found / unknown" (the bot then offers human handoff).
    A dict has: number, status (a code), location, eta, updated.

format_status(record, lang) -> str
    A friendly, bilingual one-liner for the customer.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPMENTS_PATH = os.path.join(HERE, "shipments.json")

# Human-friendly text per status code, per language.
STATUS_TEXT = {
    "created":          {"en": "Label created — awaiting pickup",
                         "sq": "Etiketa u krijua — në pritje të marrjes"},
    "picked_up":        {"en": "Picked up",
                         "sq": "U mor nga dërguesi"},
    "in_transit":       {"en": "In transit",
                         "sq": "Në transit"},
    "out_for_delivery": {"en": "Out for delivery",
                         "sq": "Doli për dorëzim"},
    "delivered":        {"en": "Delivered",
                         "sq": "U dorëzua"},
    "exception":        {"en": "Delivery issue — needs attention",
                         "sq": "Problem në dorëzim — kërkon vëmendje"},
    "returned":         {"en": "Returned to sender",
                         "sq": "U kthye te dërguesi"},
}

_EMOJI = {
    "created": "🏷️", "picked_up": "📦", "in_transit": "🚚",
    "out_for_delivery": "🛵", "delivered": "✅",
    "exception": "⚠️", "returned": "↩️",
}


def _load_shipments():
    """Load the demo shipment store. Returns {} if the file is missing."""
    try:
        with open(SHIPMENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _lookup_backend(number):
    """The pluggable backend. Swap this body for the client's real API call.

    Expected return: a dict with at least `status`; optionally `location`,
    `eta`, `updated`. Return None if the number is unknown."""
    return _load_shipments().get(number.upper())


def lookup(number):
    """Return a normalised status record for a tracking number, or None."""
    if not number:
        return None
    record = _lookup_backend(number.strip())
    if not record:
        return None
    return {
        "number": number.strip().upper(),
        "status": record.get("status", "in_transit"),
        "location": record.get("location"),
        "eta": record.get("eta"),
        "updated": record.get("updated"),
    }


def format_status(record, lang="en"):
    """A friendly one-liner describing where the shipment is."""
    code = record.get("status", "in_transit")
    label = STATUS_TEXT.get(code, STATUS_TEXT["in_transit"]).get(lang) \
        or STATUS_TEXT.get(code, STATUS_TEXT["in_transit"])["en"]
    emoji = _EMOJI.get(code, "📦")
    num = record["number"]

    if lang == "sq":
        line = f"{emoji} Dërgesa {num}: {label}."
        if record.get("location"):
            line += f" Vendndodhja e fundit: {record['location']}."
        if record.get("eta") and code not in ("delivered", "returned"):
            line += f" Pritet të dorëzohet: {record['eta']}."
        if code == "delivered" and record.get("updated"):
            line += f" Dorëzuar më {record['updated']}."
        return line

    line = f"{emoji} Shipment {num}: {label}."
    if record.get("location"):
        line += f" Last seen: {record['location']}."
    if record.get("eta") and code not in ("delivered", "returned"):
        line += f" Estimated delivery: {record['eta']}."
    if code == "delivered" and record.get("updated"):
        line += f" Delivered on {record['updated']}."
    return line

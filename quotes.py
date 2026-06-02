"""
quotes.py — shipping price estimator.

Computes a real, explainable estimate from a rate card (rates.json) using the
parcel weight and the destination zone. The bot can quote actual numbers
instead of hand-waving — and the same rate card is fed to the LLM so it quotes
consistently.

Public API:
  estimate(weight_kg, zone) -> dict           # price breakdown
  guess_zone(destination_text) -> zone | None # map a place name to a zone
  parse_weight(text) -> float | None          # pull "5 kg" out of a sentence
  rate_card_text(lang) -> str                 # human-readable card for grounding
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RATES_PATH = os.path.join(HERE, "rates.json")

_CURRENCY = "EUR"


def _load_rates():
    try:
        with open(RATES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"currency": _CURRENCY, "zones": {}}


def guess_zone(destination_text):
    """Map a free-text destination to a zone key, or None if unrecognised."""
    if not destination_text:
        return None
    low = destination_text.lower()
    rates = _load_rates()
    for zone_key, zone in rates.get("zones", {}).items():
        for kw in zone.get("match", []):
            if kw.lower() in low:
                return zone_key
    return None


def parse_weight(text):
    """Extract a weight in kg from text: '5 kg', '2.5kg', '500 g'. Returns
    a float in kg, or None."""
    if not text:
        return None
    low = text.lower()
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(kg|kgs|kilo|kilogram|kilograme)\b", low)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(g|gram|grame)\b", low)
    if m:
        return round(float(m.group(1).replace(",", ".")) / 1000.0, 3)
    return None


def estimate(weight_kg, zone):
    """Return a price breakdown for a parcel, or None if the zone is unknown.

    {zone, currency, weight_kg, base, per_kg, billable_kg, total}"""
    rates = _load_rates()
    zinfo = rates.get("zones", {}).get(zone)
    if not zinfo:
        return None
    currency = rates.get("currency", _CURRENCY)
    weight = max(float(weight_kg or 0), 0)
    included = zinfo.get("included_kg", 1)
    base = zinfo.get("base", 0)
    per_kg = zinfo.get("per_kg", 0)
    # Round weight up to the next kg for billing (standard courier practice).
    billable_extra = max(0, _ceil(weight) - included)
    total = round(base + billable_extra * per_kg, 2)
    return {
        "zone": zone,
        "zone_label": zinfo.get("label", zone),
        "currency": currency,
        "weight_kg": weight,
        "base": base,
        "per_kg": per_kg,
        "included_kg": included,
        "billable_extra_kg": billable_extra,
        "total": total,
    }


def _ceil(x):
    return int(x) if x == int(x) else int(x) + 1


def format_estimate(q, lang="en"):
    """A friendly one-liner for an estimate dict from estimate()."""
    cur = q["currency"]
    if lang == "sq":
        return (f"💰 Vlerësim për {q['weight_kg']:g} kg te {q['zone_label']}: "
                f"~{q['total']} {cur} "
                f"(bazë {q['base']} {cur} për {q['included_kg']} kg + "
                f"{q['per_kg']} {cur}/kg shtesë). "
                f"Çmimi përfundimtar konfirmohet në marrje.")
    return (f"💰 Estimate for {q['weight_kg']:g} kg to {q['zone_label']}: "
            f"~{q['total']} {cur} "
            f"(base {q['base']} {cur} for {q['included_kg']} kg + "
            f"{q['per_kg']} {cur}/kg extra). "
            f"Final price is confirmed at pickup.")


def rate_card_text(lang="en"):
    """A compact rate card to embed in the LLM grounding so it quotes the same
    numbers the calculator would."""
    rates = _load_rates()
    cur = rates.get("currency", _CURRENCY)
    rows = []
    for zinfo in rates.get("zones", {}).values():
        label = zinfo.get("label", "")
        rows.append(
            f"{label}: base {zinfo.get('base')} {cur} "
            f"(incl. {zinfo.get('included_kg')} kg), "
            f"+{zinfo.get('per_kg')} {cur}/extra kg"
        )
    return "; ".join(rows)

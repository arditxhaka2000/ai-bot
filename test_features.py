"""
test_features.py — tests for tracking, quotes, messenger I/O, and the
responder's orchestration (handoff, rate limiting, deterministic answers).

Run:  python test_features.py   (or: python -m pytest test_features.py -q)

The LLM is forced off here (LLM_PROVIDER=local) so tests never hit the network
and exercise the deterministic + local-brain paths.
"""

import config

config.LLM_PROVIDER = "local"  # force-disable the LLM for hermetic tests

import messenger  # noqa: E402
import quotes  # noqa: E402
import responder  # noqa: E402
import tracking  # noqa: E402


# --- tracking -----------------------------------------------------------------

def test_tracking_known_number():
    rec = tracking.lookup("CT123456789AL")
    assert rec and rec["status"] == "in_transit"
    assert "CT123456789AL" in tracking.format_status(rec, "en")


def test_tracking_lookup_is_case_insensitive():
    assert tracking.lookup("ct123456789al")["number"] == "CT123456789AL"


def test_tracking_unknown_number():
    assert tracking.lookup("ZZ000000000ZZ") is None


def test_tracking_status_bilingual():
    rec = tracking.lookup("CT555000111XK")  # delivered
    assert "Delivered" in tracking.format_status(rec, "en")
    assert "dorëzua" in tracking.format_status(rec, "sq").lower()


# --- quotes -------------------------------------------------------------------

def test_parse_weight():
    assert quotes.parse_weight("send a 5 kg box") == 5.0
    assert quotes.parse_weight("2.5kg") == 2.5
    assert quotes.parse_weight("500 g") == 0.5
    assert quotes.parse_weight("no weight here") is None


def test_guess_zone():
    assert quotes.guess_zone("to Tirana") == "domestic"
    assert quotes.guess_zone("ne Prishtine") == "regional"
    assert quotes.guess_zone("to Germany") == "international"
    assert quotes.guess_zone("to the moon") is None


def test_estimate_math():
    # 5 kg domestic: base 3 (incl 2 kg) + ceil(5)-2=3 extra * 0.5 = 4.5
    q = quotes.estimate(5, "domestic")
    assert q["total"] == 4.5
    assert "EUR" in quotes.format_estimate(q, "en")
    assert "EUR" in quotes.format_estimate(q, "sq")


def test_estimate_unknown_zone():
    assert quotes.estimate(5, "mars") is None


# --- messenger I/O ------------------------------------------------------------

def test_split_long_message():
    chunks = messenger._split("word " * 1000)
    assert all(len(c) <= messenger.MAX_LEN for c in chunks)
    assert len(chunks) > 1


def test_split_short_message():
    assert messenger._split("hello") == ["hello"]
    assert messenger._split("") == []


def test_duplicate_detection():
    assert messenger.is_duplicate("mid.unique.123") is False
    assert messenger.is_duplicate("mid.unique.123") is True
    assert messenger.is_duplicate(None) is False


# --- responder orchestration --------------------------------------------------

def test_responder_tracking_number_gets_real_status():
    u = "t_track"
    responder.reset(u)
    r = responder.respond(u, "CT987654321AL")
    assert r["source"] == "tracking"
    assert "CT987654321AL" in r["reply"]
    assert "delivery" in r["reply"].lower() or "🛵" in r["reply"]


def test_responder_unknown_tracking_offers_help():
    u = "t_track2"
    responder.reset(u)
    r = responder.respond(u, "ZZ111222333ZZ")
    assert r["source"] == "tracking"
    assert "couldn't find" in r["reply"].lower()


def test_responder_instant_quote():
    u = "t_quote"
    responder.reset(u)
    r = responder.respond(u, "how much to send 5 kg to Tirana?")
    assert r["source"] == "quote"
    assert "4.5" in r["reply"] and "EUR" in r["reply"]


def test_responder_quote_in_albanian():
    u = "t_quote_sq"
    responder.reset(u)
    r = responder.respond(u, "sa kushton nje pako 3 kg ne Prishtine")
    assert r["source"] == "quote"
    assert "EUR" in r["reply"]


def test_responder_human_handoff_flow():
    u = "t_handoff"
    responder.reset(u)
    # Ask for a human -> handoff message, bot enters handoff mode.
    r1 = responder.respond(u, "I want to talk to a human")
    assert r1["source"] == "handoff" and r1["handoff"] is True
    # While handed off, the bot stays silent (lets the team reply).
    r2 = responder.respond(u, "are you there?")
    assert r2["source"] == "handoff-silent" and r2["reply"] is None
    # Customer asks for the bot back -> resumes.
    r3 = responder.respond(u, "bot")
    assert r3["source"] == "resume" and r3["handoff"] is False


def test_responder_rate_limit():
    u = "t_rate"
    responder.reset(u)
    sources = [responder.respond(u, "hi")["source"]
               for _ in range(responder.RATE_MAX + 3)]
    assert "ratelimit" in sources


def test_responder_falls_back_to_local_brain():
    u = "t_local"
    responder.reset(u)
    r = responder.respond(u, "pershendetje")
    assert r["source"].startswith("local")
    assert r["reply"]


# --- standalone runner --------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")

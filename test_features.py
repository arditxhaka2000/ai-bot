"""
test_features.py — tests for messenger I/O, persistence, and the responder's
orchestration (handoff, rate limiting, leads, escalation) for the dispatch bot.

Run:  python test_features.py   (or: python -m pytest test_features.py -q)

The LLM is forced off (LLM_PROVIDER=local) and the DB points at a temp file, so
tests never hit the network or the real database.
"""

import os
import tempfile

import config

config.LLM_PROVIDER = "local"  # force-disable the LLM for hermetic tests
# Force the LOCAL SQLite backend so tests never touch the real Turso cloud DB.
config.TURSO_DATABASE_URL = ""
config.TURSO_AUTH_TOKEN = ""
config.DB_PATH = os.path.join(tempfile.gettempdir(), "cargoteer_test.db")
for _ext in ("", "-wal", "-shm"):
    try:
        os.remove(config.DB_PATH + _ext)
    except OSError:
        pass

import messenger  # noqa: E402
import responder  # noqa: E402
import store  # noqa: E402

store.init()


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

def test_get_started_welcome_is_human_dispatch():
    u = "t_gs"
    responder.reset(u)
    r = responder.respond(u, "GET_STARTED")
    assert r["source"] == "get_started"
    assert "Cargoteer" in r["reply"]
    assert "bot" not in r["reply"].lower()


def test_human_handoff_flow():
    u = "t_handoff"
    responder.reset(u)
    r1 = responder.respond(u, "I want to talk to a real person")
    assert r1["source"] == "handoff" and r1["handoff"] is True
    r2 = responder.respond(u, "are you there?")
    assert r2["source"] == "handoff-silent" and r2["reply"] is None
    r3 = responder.respond(u, "continue")
    assert r3["source"] == "resume" and r3["handoff"] is False


def test_rate_limit():
    u = "t_rate"
    responder.reset(u)
    sources = [responder.respond(u, "hi")["source"]
               for _ in range(responder.RATE_MAX + 3)]
    assert "ratelimit" in sources


def test_falls_back_to_local_brain():
    u = "t_local"
    responder.reset(u)
    r = responder.respond(u, "what do you charge")
    assert r["source"].startswith("local")
    assert r["reply"]


# --- persistence + enrichment -------------------------------------------------

def test_history_persists():
    u = "t_persist"
    responder.reset(u)
    responder.respond(u, "hello")
    responder.respond(u, "thanks")
    hist = store.get_history(u, 10)
    assert [m["role"] for m in hist][:2] == ["user", "assistant"]
    contents = " ".join(m["content"] for m in hist)
    assert "hello" in contents and "thanks" in contents


def test_state_language_is_sticky():
    u = "t_state"
    responder.reset(u)
    responder.respond(u, "sa eshte tarifa")  # Albanian pricing question
    assert store.get_state(u)["lang"] == "sq"


def test_lead_captured_for_get_started():
    u = "t_lead"
    responder.reset(u)
    responder.respond(u, "i want to get started")
    kinds = {ld["kind"] for ld in store.recent_leads(20) if ld["sender"] == u}
    assert "get_started" in kinds


def test_lead_captured_for_competitor_pricing():
    u = "t_lead2"
    responder.reset(u)
    responder.respond(u, "another dispatcher charges 6%")
    kinds = {ld["kind"] for ld in store.recent_leads(20) if ld["sender"] == u}
    assert "competitor_pricing" in kinds


def test_contact_info_is_captured_as_lead():
    u = "t_contact"
    responder.reset(u)
    responder.respond(u, "sure, reach me at arditxhaka2000@gmail.com")
    leads = [ld for ld in store.recent_leads(20) if ld["sender"] == u]
    contact_leads = [ld for ld in leads if ld["kind"] == "contact"]
    assert contact_leads, "expected a contact lead"
    assert "arditxhaka2000@gmail.com" in contact_leads[0]["text"]


def test_phone_number_is_captured_as_lead():
    u = "t_phone"
    responder.reset(u)
    responder.respond(u, "call me at +1 312 555 0148")
    kinds = {ld["kind"] for ld in store.recent_leads(20) if ld["sender"] == u}
    assert "contact" in kinds


def test_auto_escalation_after_repeated_frustration():
    u = "t_escalate"
    responder.reset(u)
    r1 = responder.respond(u, "this is terrible and useless")
    assert r1["source"] != "escalation"
    r2 = responder.respond(u, "still terrible, worst service")
    assert r2["source"] == "escalation" and r2["handoff"] is True
    assert any(ld["kind"] == "complaint"
               for ld in store.recent_leads(20) if ld["sender"] == u)


def test_reset_clears_state():
    u = "t_reset"
    responder.respond(u, "hello")
    responder.reset(u)
    assert store.get_history(u, 10) == []
    assert store.get_state(u)["handoff"] is False


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

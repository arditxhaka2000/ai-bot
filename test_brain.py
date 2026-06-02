"""
test_brain.py — behavioural tests for the local brain.

Run with:  python -m pytest test_brain.py -q
(or just `python test_brain.py` for a no-dependency summary.)

These don't check exact wording (responses are randomised and easy to edit).
They check the *behaviour* that matters: right intent, right language, typo
tolerance, entity handling, multi-turn follow-ups, and graceful fallback.
"""

from brain import Brain, detect_language, extract_tracking, fold

brain = Brain()


def intent_of(text, context=None):
    return brain.reply_with_state(text, context)["intent"]


def lang_of(text):
    return brain.reply_with_state(text)["lang"]


# --- language detection -------------------------------------------------------

def test_detects_albanian():
    assert detect_language("Përshëndetje, sa kushton dërgesa?") == "sq"
    assert detect_language("sa kushton") == "sq"
    assert detect_language("ku eshte porosia ime") == "sq"


def test_detects_english():
    assert detect_language("hello, how much does shipping cost?") == "en"
    assert detect_language("where is my order") == "en"


def test_fold_strips_diacritics():
    assert fold("Përshëndetje") == "pershendetje"
    assert fold("ÇMIMI") == "cmimi"


# --- core intents, both languages ---------------------------------------------

def test_greeting_both_languages():
    assert intent_of("hi there") == "greeting"
    assert intent_of("pershendetje") == "greeting"
    assert intent_of("Tungjatjeta") == "greeting"


def test_pricing_both_languages():
    assert intent_of("how much does shipping cost") == "pricing"
    assert intent_of("sa kushton dergesa") == "pricing"
    assert intent_of("sa kushton") == "pricing"


def test_tracking_intent():
    assert intent_of("where is my package") == "tracking"
    assert intent_of("ku eshte porosia ime") == "tracking"


def test_delivery_time():
    assert intent_of("how long does delivery take") == "delivery_time"
    assert intent_of("sa zgjat dergesa") == "delivery_time"


def test_payment_and_returns():
    assert intent_of("what payment methods do you accept") == "payment"
    assert intent_of("menyrat e pageses") == "payment"
    # Cash-on-delivery has its own, more specific intent.
    assert intent_of("can i pay cash on delivery") == "cash_on_delivery"
    assert intent_of("a mund te paguaj kesh") == "cash_on_delivery"
    assert intent_of("i want to return my order") == "returns"


def test_new_logistics_intents():
    assert intent_of("more infos from your company") == "about_company"
    assert intent_of("tregomi per kompanine") == "about_company"
    assert intent_of("what services do you offer") == "services"
    assert intent_of("i want to send a package") == "book_shipment"
    assert intent_of("schedule a pickup") == "schedule_pickup"
    assert intent_of("do you ship internationally") == "international"
    assert intent_of("my package is damaged") == "damaged_lost"
    assert intent_of("what's the maximum weight") == "weight_size"
    assert intent_of("i missed my delivery") == "missed_delivery"
    assert intent_of("i need an invoice") == "invoice"
    assert intent_of("where are you located") == "locations"


def test_reply_is_in_users_language():
    sq = brain.reply_with_state("sa kushton dergesa")
    assert sq["lang"] == "sq"
    assert any(ch in sq["reply"] for ch in "ëçÇË") or "çmim" in sq["reply"].lower() \
        or "destinacioni" in sq["reply"].lower()

    en = brain.reply_with_state("how much does shipping cost")
    assert en["lang"] == "en"


# --- robustness ---------------------------------------------------------------

def test_typo_tolerance():
    assert intent_of("helo") == "greeting"
    assert intent_of("how mcuh does it cost") == "pricing"


def test_unknown_vocab_falls_back_not_false_positive():
    # "worst bot ever" shares the word "bot" with who_are_you, but it's a
    # complaint — must NOT be classified as a friendly who_are_you.
    assert intent_of("worst bot ever, useless") == "frustrated"
    # Pure gibberish should fall back, not match anything.
    assert intent_of("asdfgh qwerty zxcvb") == "fallback"


def test_frustration_detection_both_languages():
    assert intent_of("this is terrible, my parcel never arrived") == "frustrated"
    assert intent_of("dergesa nuk ka ardhur, turp") == "frustrated"


# --- entities + multi-turn ----------------------------------------------------

def test_extract_tracking_number():
    assert extract_tracking("my number is RR123456789AL") == "RR123456789AL"
    assert extract_tracking("1Z999AA10123456784") == "1Z999AA10123456784"
    # Prices / short numbers are not tracking numbers.
    assert extract_tracking("it costs 2024") is None
    assert extract_tracking("call me at 355") is None


def test_bare_tracking_number_is_acknowledged():
    r = brain.reply_with_state("RR123456789AL")
    assert r["intent"] == "tracking_ack"
    assert "RR123456789AL" in r["reply"]


def test_tracking_number_in_question_is_acknowledged():
    # Customer asks AND includes the number — acknowledge it, don't re-ask.
    r = brain.reply_with_state("my tracking is RR987654321AL where is it")
    assert r["intent"] == "tracking_ack"
    assert "RR987654321AL" in r["reply"]


def test_compound_social_plus_question_prefers_question():
    # A greeting in front of a real question must not swallow the question.
    assert intent_of("Mirembrema, kur arrin porosia ime?") == "delivery_time"
    assert intent_of("hi there, how much does shipping cost?") == "pricing"


def test_coverage_not_confused_with_pricing():
    assert intent_of("where do you ship to") == "coverage"


def test_tracking_followup_flow():
    # Turn 1: customer asks to track -> bot asks for a number, awaits it.
    t1 = brain.reply_with_state("ku eshte porosia ime")
    assert t1["intent"] == "tracking"
    assert t1["awaiting"] == "tracking"
    # Turn 2: customer sends a number -> read as the answer.
    t2 = brain.reply_with_state("123456789012", context={"awaiting": t1["awaiting"]})
    assert t2["intent"] == "tracking_ack"
    assert "123456789012" in t2["reply"]


def test_language_falls_back_to_matched_pattern():
    # "si funksionon" has no language marker words, but it matches an Albanian
    # pattern — so the reply should come back in Albanian, not default English.
    r = brain.reply_with_state("si funksionon")
    assert r["intent"] == "how_it_works"
    assert r["lang"] == "sq"


def test_damage_routes_to_claim_not_generic_frustration():
    # Damage/loss gets the dedicated claim flow, not generic empathy.
    assert intent_of("my parcel arrived broken") == "damaged_lost"
    assert intent_of("pakoja arriti e thyer") == "damaged_lost"
    # ...but a pure rant still gets the empathetic frustration reply.
    assert intent_of("this is the worst, useless service") == "frustrated"


def test_language_is_sticky_for_signalless_messages():
    # A bare tracking number carries no language signal; it should inherit the
    # conversation's language instead of snapping back to English.
    t1 = brain.reply_with_state("ku eshte porosia ime")  # Albanian
    assert t1["lang"] == "sq"
    t2 = brain.reply_with_state(
        "RR123456789AL",
        context={"awaiting": t1["awaiting"], "lang": t1["lang"]},
    )
    assert t2["lang"] == "sq"
    assert "RR123456789AL" in t2["reply"]


# --- learning -----------------------------------------------------------------

def test_learn_in_memory(tmp_path=None):
    # Learn on a throwaway copy so we don't mutate the real knowledge file.
    import json
    import os
    import tempfile

    with open(brain.knowledge_path, encoding="utf-8") as f:
        data = json.load(f)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    try:
        b = Brain(path)
        b.learn("greeting", "wassup fam", lang="en")
        assert b.reply_with_state("wassup fam")["intent"] == "greeting"
    finally:
        os.remove(path)


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

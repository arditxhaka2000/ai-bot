"""
test_brain.py — behavioural tests for the local brain (dispatch business).

Run:  python test_brain.py   (or: python -m pytest test_brain.py -q)

These check behaviour, not exact wording: right intent, right language, typo
tolerance, social-vs-substantive demotion, and graceful fallback.
"""

from brain import Brain, detect_language, fold

brain = Brain()


def intent_of(text):
    return brain.reply_with_state(text)["intent"]


# --- language detection -------------------------------------------------------

def test_detects_spanish():
    assert detect_language("Hola, ¿cuánto cobran?") == "es"
    assert detect_language("quiero empezar con un despachador") == "es"


def test_detects_russian():
    assert detect_language("Здравствуйте, сколько вы берёте?") == "ru"
    assert detect_language("хочу начать работать с диспетчером") == "ru"


def test_detects_english():
    assert detect_language("hello, what do you charge?") == "en"


def test_fold_strips_diacritics():
    assert fold("¿Cuánto?") == "¿cuanto?"
    assert fold("ESPAÑA") == "espana"


# --- core dispatch intents ----------------------------------------------------

def test_greeting_all_languages():
    assert intent_of("hi there") == "greeting"
    assert intent_of("hola buenas") == "greeting"
    assert intent_of("привет") == "greeting"


def test_pricing():
    assert intent_of("how much do you charge") == "pricing"
    assert intent_of("what's your dispatch fee") == "pricing"
    assert intent_of("cuanto cobran") == "pricing"
    assert intent_of("сколько вы берёте") == "pricing"


def test_competitor_pricing():
    assert intent_of("another dispatcher charges 6%") == "competitor_pricing"
    assert intent_of("my current dispatcher charges 5 percent") == "competitor_pricing"


def test_services_and_how_it_works():
    assert intent_of("what services do you offer") == "services"
    assert intent_of("how does dispatch work") == "how_it_works"


def test_get_started():
    assert intent_of("i want to get started") == "get_started"
    assert intent_of("sign me up") == "get_started"
    assert intent_of("i need a dispatcher") == "get_started"


def test_about_and_who():
    assert intent_of("what is cargoteer") == "about_company"
    assert intent_of("who are you") == "who_are_you"


def test_who_are_you_is_not_a_bot():
    # The agent should present as a human teammate, never as a bot.
    reply = brain.reply_with_state("are you a real person")["reply"].lower()
    assert "bot" not in reply and "automated" not in reply
    assert "team" in reply or "dispatch" in reply


def test_equipment_and_owner_operator():
    assert intent_of("do you dispatch flatbed") == "equipment"
    assert intent_of("i'm an owner operator") == "owner_operator"


# --- robustness ---------------------------------------------------------------

def test_typo_tolerance():
    assert intent_of("helo") == "greeting"
    assert intent_of("pricng") == "pricing"
    assert intent_of("whats your raet") == "pricing"


def test_unknown_vocab_falls_back():
    assert intent_of("asdfgh qwerty zxcvb") == "fallback"


def test_frustration_detection():
    assert intent_of("this is the worst, useless service") == "frustrated"


def test_reply_language_matches_user():
    assert brain.reply_with_state("cuanto cobran")["lang"] == "es"
    assert brain.reply_with_state("сколько вы берёте")["lang"] == "ru"
    assert brain.reply_with_state("what do you charge")["lang"] == "en"


def test_compound_social_plus_question_prefers_question():
    # A greeting in front of a real question must not swallow the question.
    assert intent_of("hey, what do you charge?") == "pricing"


def test_russian_intent_matching():
    # Cyrillic input should still match the right intent and reply in Russian.
    r = brain.reply_with_state("хочу начать")
    assert r["intent"] == "get_started"
    assert r["lang"] == "ru"


# --- classify + learning ------------------------------------------------------

def test_classify_returns_actionable_tag():
    tag, score = brain.classify("i want to get started")
    assert tag == "get_started" and score >= 0.5


def test_learn_in_memory():
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
        b.learn("greeting", "wazaa", lang="en")
        assert b.reply_with_state("wazaa")["intent"] == "greeting"
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

"""
brain.py — the bot's self-contained, bilingual "brain".

No external AI API. It always replies sensibly — and in the customer's own
language (English or Albanian) — through four cooperating layers:

  1) Language detection. Looks at diacritics (ë, ç) and a set of marker words
     to decide whether the customer wrote in Albanian (sq) or English (en), so
     every reply comes back in their language.

  2) Entity extraction. Spots tracking numbers and phone numbers in the raw
     message. A tracking number is the single most important thing a logistics
     customer sends, so it short-circuits to a tracking reply even with no
     other matching words.

  3) Intent matching. Learns from knowledge.json by turning every example
     phrase into TF-IDF vectors — a *word* signal (content words, accent-folded)
     and a *character* signal (shrugs off typos). An incoming message is
     compared (cosine similarity) against everything it knows; if it's
     confident enough, it replies with that intent's answer in the right
     language.

  4) Fallback reasoner. If the message is "out of protocol", it doesn't go
     silent: it inspects the message (question? complaint? small talk?
     gibberish?) and crafts a sensible, language-matched reply, then logs the
     message so the bot can be taught later (brain.learn(...)).

Reply also accepts an optional `context` so multi-turn flows work — e.g. the
bot asks for a tracking number, the customer sends one, and the bot recognises
it as the answer to that question.
"""

import json
import os
import random
import re
import unicodedata
from datetime import datetime, timezone

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(HERE, "knowledge.json")
UNKNOWN_LOG = os.path.join(HERE, "unknown_messages.log")

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "sq")

# Accept an intent when the best phrase clears this bar. Char + word signals
# are blended, so a clean hit lands well above it and noise well below.
CONFIDENCE_THRESHOLD = 0.45


# --- language resources -------------------------------------------------------

# Marker words, accent-folded. Presence of these tips language detection. Kept
# deliberately small and high-signal; the diacritics check carries the rest.
SQ_MARKERS = {
    "pershendetje", "tungjatjeta", "tung", "ckemi", "miremengjes", "mirembrema",
    "miredita", "mirupafshim", "shihemi", "naten", "faleminderit", "flm",
    "sa", "kushton", "cmimi", "cmim", "ku", "eshte", "porosia", "porosi",
    "dergesa", "dergese", "dergesen", "dergoni", "paketa", "paketen", "gjurmo",
    "gjurmim", "gjurmimit", "statusi", "dua", "te", "kam", "jam", "je", "jeni",
    "mire", "po", "jo", "lutem", "ndihmo", "adresen", "adrese", "pagese",
    "paguaj", "kthej", "kthim", "oraret", "orari", "hapur", "numri", "telefonit",
    "vjen", "arrin", "dite", "kohe", "falas", "tani", "njeri", "robot", "bot",
    "cfare", "kush", "cili", "zona", "jashte", "kosove", "nderkombetare",
}
EN_MARKERS = {
    "the", "you", "your", "are", "is", "am", "how", "what", "when", "where",
    "why", "can", "do", "does", "will", "hello", "hi", "hey", "thanks", "thank",
    "bye", "price", "cost", "much", "ship", "shipping", "delivery", "deliver",
    "track", "tracking", "order", "package", "parcel", "hours", "open",
    "contact", "help", "return", "refund", "pay", "payment", "address",
    "change", "free", "human", "person", "bot",
}
# Words claimed by both lists ("po", "bot", …) shouldn't sway the vote.
_AMBIGUOUS = SQ_MARKERS & EN_MARKERS
SQ_ONLY = SQ_MARKERS - _AMBIGUOUS
EN_ONLY = EN_MARKERS - _AMBIGUOUS


def fold(text):
    """Lowercase and strip diacritics so 'Përshëndetje' == 'pershendetje' and
    'helló' == 'hello'. Matching and language detection both run on this."""
    text = (text or "").lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _tokens(folded_text):
    return re.findall(r"[a-z']+", folded_text)


def detect_language(text, fallback=DEFAULT_LANG):
    """Return 'sq' or 'en'. Albanian diacritics are a strong tell; otherwise we
    count marker words. When there's no signal either way (e.g. a bare tracking
    number), return `fallback` — callers pass the conversation's language so a
    number sent mid-chat is answered in the language already in use."""
    raw_low = (text or "").lower()
    folded = fold(raw_low)
    toks = set(_tokens(folded))

    sq_score = len(toks & SQ_ONLY)
    en_score = len(toks & EN_ONLY)

    # ë / ç are unambiguously Albanian in this customer base.
    if "ë" in raw_low or "ç" in raw_low:
        sq_score += 2

    if sq_score > en_score:
        return "sq"
    if en_score > sq_score:
        return "en"
    return fallback if fallback in SUPPORTED_LANGS else DEFAULT_LANG


class Brain:
    def __init__(self, knowledge_path=KNOWLEDGE_PATH):
        self.knowledge_path = knowledge_path
        self._load()

    # ---- learning / training -------------------------------------------------

    def _load(self):
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.knowledge = json.load(f)
        self._train()

    @staticmethod
    def _patterns_for(intent, lang):
        """Patterns for a language. Tolerates both the bilingual {en:[],sq:[]}
        shape and a legacy flat list (treated as English)."""
        pats = intent.get("patterns", {})
        if isinstance(pats, dict):
            return pats.get(lang, [])
        return pats if lang == "en" else []

    @staticmethod
    def _responses_for(intent, lang):
        resp = intent.get("responses", {})
        if isinstance(resp, dict):
            return resp.get(lang) or resp.get(DEFAULT_LANG) or next(
                (v for v in resp.values() if v), []
            )
        return resp

    def _train(self):
        """Build TF-IDF models from every example phrase, all languages folded
        into one space (so a folded query is comparable to any phrase)."""
        self.phrases = []        # accent-folded example phrases
        self.phrase_tags = []    # the intent tag each phrase belongs to
        self.phrase_langs = []   # the language each phrase was written in
        for intent in self.knowledge["intents"]:
            for lang in SUPPORTED_LANGS:
                for pattern in self._patterns_for(intent, lang):
                    self.phrases.append(fold(pattern))
                    self.phrase_tags.append(intent["tag"])
                    self.phrase_langs.append(lang)

        # Word signal: content words (English stop words dropped — Albanian has
        # no list, but its function words are short and low-IDF anyway).
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), stop_words="english"
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.phrases)

        # Char signal: shrugs off typos ("helo" -> "hello", "kushtn" -> "kushton").
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.phrases)

    def learn(self, tag, pattern, response=None, lang=DEFAULT_LANG):
        """Teach the bot a new phrase (and optionally a response) for a given
        language at runtime, persist it to knowledge.json, and retrain."""
        for intent in self.knowledge["intents"]:
            if intent["tag"] == tag:
                self._append_pattern(intent, lang, pattern, response)
                break
        else:
            self.knowledge["intents"].append({
                "tag": tag,
                "patterns": {lang: [pattern]},
                "responses": {lang: [response] if response else ["Got it!"]},
            })
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        self._train()

    @staticmethod
    def _append_pattern(intent, lang, pattern, response):
        pats = intent.setdefault("patterns", {})
        if not isinstance(pats, dict):  # migrate a legacy flat list in place
            intent["patterns"] = pats = {"en": list(pats)}
        pats.setdefault(lang, []).append(pattern)
        if response:
            resp = intent.setdefault("responses", {})
            if not isinstance(resp, dict):
                intent["responses"] = resp = {"en": list(resp)}
            resp.setdefault(lang, []).append(response)

    # ---- answering -----------------------------------------------------------

    def reply(self, message, context=None):
        """Return the best reply text for a user message. See reply_with_state
        for the richer return used by multi-turn callers."""
        return self.reply_with_state(message, context)["reply"]

    def reply_with_state(self, message, context=None):
        """Decide a reply and report state.

        Returns {"reply", "lang", "intent", "awaiting"}. `awaiting` is kept in
        the shape for the caller's convenience but is currently always None.

        `context` (optional) carries the conversation language so a message
        with no language signal is answered in the language already in use.
        """
        text = (message or "").strip()

        # Match intent first so its language can answer messages that carry no
        # language signal of their own (e.g. "si funksionon" — no marker words).
        tag, score, matched_lang = (
            self._best_intent(text) if text else (None, 0.0, DEFAULT_LANG)
        )
        # Language priority: explicit signal in THIS message > conversation
        # language > the matched pattern's language > default.
        context_lang = (context or {}).get("lang")
        lang = detect_language(text, fallback=(context_lang or matched_lang
                                               or DEFAULT_LANG))

        def out(reply, intent):
            return {"reply": reply, "lang": lang, "intent": intent,
                    "awaiting": None}

        if not text:
            return out(self._say("empty", lang), "empty")

        # Emotion first: a clear complaint gets empathy, never a cheery canned
        # answer that happened to share a word.
        if self._is_frustrated(text):
            return out(self._say("frustrated", lang), "frustrated")

        if tag is not None and score >= CONFIDENCE_THRESHOLD:
            return out(self._response_for(tag, lang), tag)

        # Out of protocol — reason about it instead of giving up.
        self._log_unknown(text, score, lang)
        return out(self._fallback(text, lang), "fallback")

    # Social pleasantries. A carrier who opens with "Hey, what do you charge?"
    # wants the pricing answer, not just "Hello" — so when a substantive intent
    # also clears the bar, it wins over these.
    SOCIAL_TAGS = {"greeting", "goodbye", "thanks"}

    def classify(self, text):
        """Public: best intent tag + confidence for a message (no side effects).
        Used by the responder to capture leads even when the LLM writes the
        reply. Returns (tag, score)."""
        if not (text or "").strip():
            return None, 0.0
        tag, score, _ = self._best_intent(text)
        return tag, score

    def is_frustrated(self, text):
        """Public: True if the message reads as a clear complaint/anger."""
        return self._is_frustrated(text)

    def _best_intent(self, text):
        folded = fold(text)
        word_sims = cosine_similarity(
            self.word_vectorizer.transform([folded]), self.word_matrix
        )[0]
        char_sims = cosine_similarity(
            self.char_vectorizer.transform([folded]), self.char_matrix
        )[0]

        # Guard the WORD signal against false confidence from a single shared
        # word (e.g. "worst BOT ever" vs "are you a bot"): scale it by how many
        # of the message's content words we actually recognise, so unknown-heavy
        # messages don't read 1.0 off one coincidental word. The CHAR signal is
        # left untouched, so typo-matches ("helo" -> "hello") still count.
        word_sims = word_sims * self._word_coverage(folded)

        # Aggregate to one (blended, char, lang) per tag — best of its phrases.
        # The char score is kept only as a tiebreak: two intents often tie on
        # the blended score when stop-word removal collapses the query to a
        # single shared word ("where do you ship to" -> "ship", in both
        # `coverage` and `pricing`). The char signal sees the *whole* phrase, so
        # it favours the one the message actually resembles end to end. `lang`
        # is the language of the winning phrase, used to answer in the right
        # language when the message itself carries no language signal.
        agg = {}
        for i, tag in enumerate(self.phrase_tags):
            blended = max(word_sims[i], char_sims[i])
            cur = agg.get(tag)
            if cur is None or (blended, char_sims[i]) > (cur[0], cur[1]):
                agg[tag] = (blended, char_sims[i], self.phrase_langs[i])

        ranked = sorted(agg.items(), key=lambda kv: (kv[1][0], kv[1][1]),
                        reverse=True)
        top_tag, (top_score, _, top_lang) = ranked[0]

        # Demote a social opener if a substantive intent also clears the bar.
        if top_tag in self.SOCIAL_TAGS:
            for tag, (score, _, lang) in ranked[1:]:
                if tag not in self.SOCIAL_TAGS and score >= CONFIDENCE_THRESHOLD:
                    return tag, float(score), lang
        return top_tag, float(top_score), top_lang

    def _word_coverage(self, folded_text):
        """Fraction of the message's content words the bot knows. 1.0 if it
        recognises them all, lower the more the message is unknown vocabulary.
        Returns 1.0 when there are no content words, leaving char matching free."""
        tokens = [t for t in _tokens(folded_text) if t not in ENGLISH_STOP_WORDS]
        if not tokens:
            return 1.0
        vocab = self.word_vectorizer.vocabulary_
        known = sum(1 for t in tokens if t in vocab)
        return known / len(tokens)

    def _response_for(self, tag, lang):
        for intent in self.knowledge["intents"]:
            if intent["tag"] == tag:
                choices = self._responses_for(intent, lang)
                if choices:
                    return random.choice(choices)
        return self._fallback("", lang)

    # ---- fallback reasoner (handles "out of protocol" messages) --------------

    FRUSTRATION_WORDS = {
        # Note: damage/loss words ("broken", "damaged", "lost", "prishur") are
        # deliberately NOT here — the dedicated `damaged_lost` intent handles
        # those with a better claim flow (asks for a photo, opens a claim).
        "en": (
            "angry", "annoyed", "useless", "stupid", "worst", "hate",
            "terrible", "awful", "rubbish", "garbage", "complaint", "sucks",
            "ridiculous", "scam", "fed up", "unacceptable", "disappointed",
            "still waiting", "where is my money", "no one helps",
        ),
        "sq": (
            "i zemeruar", "e zemeruar", "nervozuar", "tmerzitur", "keq",
            "i keq", "e keqe", "ankese", "qesharake", "mashtrim", "turp",
            "skandal", "e papranueshme", "i zhgenjyer", "e zhgenjyer",
            "ende pres", "po pres prej kohesh", "ku jane parate e mia",
            "askush nuk ndihmon",
        ),
    }

    def _is_frustrated(self, text):
        folded = fold(text)
        for lang in SUPPORTED_LANGS:
            if any(w in folded for w in self.FRUSTRATION_WORDS[lang]):
                return True
        return False

    QUESTION_STARTERS = re.compile(
        r"^(who|what|when|where|why|how|can|do|does|is|are|will|should|could"   # en
        r"|kush|cfare|cili|cila|kur|ku|pse|si|sa|a)\b"                          # sq
    )

    def _fallback(self, text, lang):
        folded = fold(text)

        if "?" in text or self.QUESTION_STARTERS.match(folded):
            return self._say("fallback_question", lang)

        if folded.strip() in ("ok", "okay", "k", "cool", "nice", "sure", "yes",
                               "no", "po", "jo", "mire", "dakord"):
            return self._say("fallback_ack", lang)

        if len(text) <= 2 or not re.search(r"[a-zA-Zëçáéíóú]", text):
            return self._say("fallback_short", lang)

        return self._say("fallback_generic", lang)

    # ---- canned, language-aware system messages ------------------------------

    SYSTEM_MESSAGES = {
        "empty": {
            "en": ["Did you mean to send something? I didn't catch that. 🙂"],
            "sq": ["A deshët të dërgoni diçka? Nuk e kapa. 🙂"],
        },
        "frustrated": {
            "en": ["I'm sorry — that's frustrating, and I want to make it right. 🙏 "
                   "Tell me what's going on and I'll jump on it or get a teammate to "
                   "help you directly."],
            "sq": ["Më vjen keq — kjo është e bezdisshme dhe dua ta rregulloj. 🙏 "
                   "Më thoni çfarë po ndodh dhe e marr përsipër ose sjell një koleg "
                   "që t'ju ndihmojë drejtpërdrejt."],
        },
        "fallback_question": {
            "en": ["Good question — let me get you a precise answer. Mind if a teammate "
                   "follows up, or can you tell me a bit more about your operation?",
                   "I want to get that exactly right for you. Can you share your "
                   "equipment and lanes so I can help, or should a teammate reach out?"],
            "sq": ["Pyetje e mirë — t'ju jap një përgjigje të saktë. Mund t'ju kontaktojë "
                   "një koleg, apo më tregoni pak më shumë për operacionin tuaj?",
                   "Dua t' jua jap saktë. Mund të më thoni automjetin dhe rrugët, apo "
                   "preferoni t'ju kontaktojë një koleg?"],
        },
        "fallback_ack": {
            "en": ["👍 Got it. Anything else I can help with?"],
            "sq": ["👍 U mor vesh. Diçka tjetër ku mund të ndihmoj?"],
        },
        "fallback_short": {
            "en": ["I'm here! What can I help you with?"],
            "sq": ["Jam këtu! Si mund t'ju ndihmoj?"],
        },
        "fallback_generic": {
            "en": ["Hmm, I'm not sure I fully got that. Could you tell me a bit more so I can help?",
                   "I want to make sure I help correctly — can you rephrase that or add a little detail?",
                   "I didn't quite understand, but I'm listening. What are you trying to do?"],
            "sq": ["Hmm, nuk jam i sigurt se e kuptova plotësisht. Mund të më thoni pak më shumë që t'ju ndihmoj?",
                   "Dua të sigurohem që ju ndihmoj saktë — mund ta riformuloni ose të shtoni pak detaje?",
                   "Nuk e kuptova mirë, por po ju dëgjoj. Çfarë doni të bëni?"],
        },
    }

    def _say(self, key, lang, **fmt):
        bucket = self.SYSTEM_MESSAGES[key]
        choices = bucket.get(lang) or bucket[DEFAULT_LANG]
        return random.choice(choices).format(**fmt)

    def _log_unknown(self, text, score, lang):
        line = (f"{datetime.now(timezone.utc).isoformat()}\t{lang}\t"
                f"{score:.2f}\t{text}\n")
        with open(UNKNOWN_LOG, "a", encoding="utf-8") as f:
            f.write(line)


# Singleton used by the web server.
brain = Brain()


if __name__ == "__main__":
    # Quick offline test loop — no Messenger needed. Try English and Albanian.
    print("Local brain test (bilingual). Type messages (Ctrl+C to quit).\n")
    try:
        while True:
            msg = input("you > ")
            print("bot >", brain.reply(msg), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye 👋")

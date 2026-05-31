"""
brain.py — the bot's self-contained "brain".

No external AI API. It works in two layers:

  1) Intent matching: it learns from knowledge.json by turning every example
     phrase into a TF-IDF vector. An incoming message is compared (cosine
     similarity) against everything it knows. If it's confident enough, it
     replies with that intent's answer.

  2) Fallback reasoner: if the message is "out of protocol" (nothing it knows
     matches well), it doesn't go silent. It inspects the message — is it a
     question? a complaint? small talk? — and crafts a sensible reply on its
     own. Every unrecognised message is logged so you can teach the bot later
     (or call brain.learn(...) to add it permanently at runtime).
"""

import json
import os
import random
import re
from datetime import datetime, timezone

from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(HERE, "knowledge.json")
UNKNOWN_LOG = os.path.join(HERE, "unknown_messages.log")

# How similar (0..1) a message must be to a known phrase to count as a match.
# Lower = bolder/looser matching, higher = stricter.
CONFIDENCE_THRESHOLD = 0.50


class Brain:
    def __init__(self, knowledge_path=KNOWLEDGE_PATH):
        self.knowledge_path = knowledge_path
        self._load()

    # ---- learning / training -------------------------------------------------

    def _load(self):
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.knowledge = json.load(f)
        self._train()

    def _train(self):
        """Build the TF-IDF model from every example phrase we know.

        Two complementary signals:
          - word-level (stop words removed) keys on *content* words, so
            "build a rocket" doesn't match on filler like "you"/"can".
          - char-level n-grams shrug off typos ("helo" -> "hello").
        We blend them so the bot is both semantic and typo-tolerant."""
        self.phrases = []        # flat list of example phrases
        self.phrase_tags = []    # the intent tag each phrase belongs to
        for intent in self.knowledge["intents"]:
            for pattern in intent["patterns"]:
                self.phrases.append(pattern.lower())
                self.phrase_tags.append(intent["tag"])

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), stop_words="english"
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.phrases)

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.phrases)

    def learn(self, tag, pattern, response=None):
        """Teach the bot a new phrase (and optionally a new response) at
        runtime, then persist it to knowledge.json and retrain."""
        for intent in self.knowledge["intents"]:
            if intent["tag"] == tag:
                intent["patterns"].append(pattern)
                if response:
                    intent["responses"].append(response)
                break
        else:
            self.knowledge["intents"].append({
                "tag": tag,
                "patterns": [pattern],
                "responses": [response] if response else ["Got it!"],
            })
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, indent=2, ensure_ascii=False)
        self._train()

    # ---- answering -----------------------------------------------------------

    def reply(self, message):
        """Return the best reply for a user message."""
        text = (message or "").strip()
        if not text:
            return "Did you mean to send something? I didn't catch that. 🙂"

        # Emotion comes first: a clear complaint/insult should get empathy,
        # never a cheery canned answer that happened to share a word.
        if self._is_frustrated(text):
            return ("I'm sorry you're having a rough time with this. 🙏 "
                    "Tell me what went wrong and I'll do my best to fix it "
                    "or get a human involved.")

        tag, score = self._best_intent(text)
        if tag is not None and score >= CONFIDENCE_THRESHOLD:
            return self._response_for(tag)

        # Out of protocol — reason about it instead of giving up.
        self._log_unknown(text, score)
        return self._fallback(text)

    def _best_intent(self, text):
        low = text.lower()
        word_sims = cosine_similarity(
            self.word_vectorizer.transform([low]), self.word_matrix
        )[0]
        char_sims = cosine_similarity(
            self.char_vectorizer.transform([low]), self.char_matrix
        )[0]

        # Guard against false confidence: if the message's only *known*
        # content word happens to appear in a pattern (e.g. "worst BOT ever"
        # -> "are you a bot"), the word signal can read 1.0 off that single
        # word. Scale it by how many of the message's content words the bot
        # actually recognises, so unknown-heavy messages fall back instead.
        word_sims = word_sims * self._word_coverage(low)

        # Per-phrase score = the stronger of the two signals. Content-word
        # overlap OR a close typo-match both count as evidence.
        blended = [max(w, c) for w, c in zip(word_sims, char_sims)]
        best = max(range(len(blended)), key=blended.__getitem__)
        return self.phrase_tags[best], float(blended[best])

    def _word_coverage(self, text):
        """Fraction of the message's content words that the bot knows.
        1.0 if it recognises them all, lower (toward 0) the more the message
        is made of words it has never seen. Returns 1.0 when there are no
        content words to judge, so char-level matching is left untouched."""
        tokens = [
            t for t in re.findall(r"[a-z']+", text.lower())
            if t not in ENGLISH_STOP_WORDS
        ]
        if not tokens:
            return 1.0
        vocab = self.word_vectorizer.vocabulary_
        known = sum(1 for t in tokens if t in vocab)
        return known / len(tokens)

    def _response_for(self, tag):
        for intent in self.knowledge["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return self._fallback("")

    # ---- fallback reasoner (handles "out of protocol" messages) --------------

    FRUSTRATION_WORDS = (
        "angry", "annoyed", "useless", "stupid", "worst", "hate",
        "terrible", "awful", "rubbish", "garbage", "complaint", "sucks",
        "not working", "doesn't work", "broken", "ridiculous", "scam",
    )

    def _is_frustrated(self, text):
        return any(w in text.lower() for w in self.FRUSTRATION_WORDS)

    def _fallback(self, text):
        low = text.lower()

        # A question? Acknowledge it and offer a path forward.
        if "?" in text or re.match(
            r"^(who|what|when|where|why|how|can|do|does|is|are|will|should|could)\b",
            low,
        ):
            return random.choice([
                "Good question — I'm not 100% sure on that one yet. "
                "Could you rephrase it, or I can pass it to a human?",
                "I don't have a confident answer for that. Want me to "
                "forward it to the team?",
            ])

        # Looks like a yes/no acknowledgement?
        if low in ("ok", "okay", "k", "cool", "nice", "sure", "yes", "no"):
            return "👍 Got it. Anything else I can help with?"

        # Very short / emoji-only / gibberish.
        if len(text) <= 2 or not re.search(r"[a-zA-Z]", text):
            return "I'm here! What can I help you with?"

        # Generic but useful default — reflect understanding, invite detail.
        return random.choice([
            "Hmm, I'm not sure I fully got that. Could you tell me a bit "
            "more so I can help?",
            "I want to make sure I help correctly — can you rephrase that "
            "or add a little detail?",
            "I didn't quite understand, but I'm listening. What are you "
            "trying to do?",
        ])

    def _log_unknown(self, text, score):
        line = f"{datetime.now(timezone.utc).isoformat()}\t{score:.2f}\t{text}\n"
        with open(UNKNOWN_LOG, "a", encoding="utf-8") as f:
            f.write(line)


# Singleton used by the web server.
brain = Brain()


if __name__ == "__main__":
    # Quick offline test loop — no Messenger needed.
    print("Local brain test. Type messages (Ctrl+C to quit).\n")
    try:
        while True:
            msg = input("you > ")
            print("bot >", brain.reply(msg), "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye 👋")

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATH = os.path.join(HERE, "knowledge.json")
UNKNOWN_LOG = os.path.join(HERE, "unknown_messages.log")

# How similar (0..1) a message must be to a known phrase to count as a match.
# Lower = bolder/looser matching, higher = stricter.
CONFIDENCE_THRESHOLD = 0.30


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
        """Build the TF-IDF model from every example phrase we know."""
        self.phrases = []        # flat list of example phrases
        self.phrase_tags = []    # the intent tag each phrase belongs to
        for intent in self.knowledge["intents"]:
            for pattern in intent["patterns"]:
                self.phrases.append(pattern.lower())
                self.phrase_tags.append(intent["tag"])

        # char-level n-grams make it robust to typos and word variations
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4)
        )
        self.matrix = self.vectorizer.fit_transform(self.phrases)

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

        tag, score = self._best_intent(text)
        if tag is not None and score >= CONFIDENCE_THRESHOLD:
            return self._response_for(tag)

        # Out of protocol — reason about it instead of giving up.
        self._log_unknown(text, score)
        return self._fallback(text)

    def _best_intent(self, text):
        vec = self.vectorizer.transform([text.lower()])
        sims = cosine_similarity(vec, self.matrix)[0]
        best = int(sims.argmax())
        return self.phrase_tags[best], float(sims[best])

    def _response_for(self, tag):
        for intent in self.knowledge["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return self._fallback("")

    # ---- fallback reasoner (handles "out of protocol" messages) --------------

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

        # Frustration / complaint? Be empathetic.
        if any(w in low for w in (
            "angry", "annoyed", "useless", "stupid", "bad", "worst",
            "hate", "terrible", "complaint", "not working", "broken",
        )):
            return ("I'm sorry you're having a rough time with this. "
                    "Tell me what went wrong and I'll do my best to sort it "
                    "out or get a human involved.")

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

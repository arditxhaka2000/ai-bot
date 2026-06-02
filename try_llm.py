"""
try_llm.py — quick check that the LLM brain is wired up and answering.

1) Put your key in .env:   GEMINI_API_KEY=...   (get one free at
   https://aistudio.google.com/app/apikey)
2) Run:  python try_llm.py

It sends a few English + Albanian messages — including an off-topic one and a
question we never pre-wrote — so you can see it formulate fresh, on-brand,
language-matched replies. If no key is set, it tells you and exits.
"""

import llm

PROBES = [
    "Hi, do you deliver fragile glass items and how should I pack them?",
    "Përshëndetje, a mund të dërgoj një dhuratë surprizë pa e ditur marrësi çmimin?",
    "what's the capital of France?",          # off-topic — should steer back
    "sa kushton me dergu nje pako 5 kg ne Prishtine?",
]


def main():
    provider = llm.active_provider()
    if provider is None:
        print("No LLM key set. Add GEMINI_API_KEY (or OPENAI_API_KEY) to .env.")
        print("Get a free Gemini key: https://aistudio.google.com/app/apikey")
        return

    print(f"Using provider: {provider}\n" + "=" * 60)
    for msg in PROBES:
        reply = llm.generate_reply([{"role": "user", "content": msg}])
        print(f"\nUSER: {msg}")
        if reply is None:
            print(" LLM: (failed — would fall back to local brain; "
                  "check the [llm] error above)")
        else:
            print(f" LLM: {reply}")


if __name__ == "__main__":
    main()

# ai-bot

A Facebook **Messenger** bot in Python with its own **self-contained,
bilingual brain** — no Claude, no OpenAI, no paid API required. It learns from a
knowledge base and reasons about messages it has never seen before, so it always
replies sensibly — and in the customer's own language — even when someone goes
"out of protocol". (When an OpenAI key *is* set, the smart brain answers first
and this brain is the free, always-on fallback.)

## How the brain works

Four cooperating layers (see `brain.py`):

- **Language detection** — decides whether the customer wrote in **English** or
  **Albanian** (diacritics like `ë`/`ç` plus marker words) and replies in that
  language. Language is *sticky* across a conversation, so a bare tracking
  number sent mid-chat is still answered in the language already in use.
- **Entity extraction** — spots **tracking numbers** and phone numbers in the
  raw message. A tracking number is the highest-value thing a logistics
  customer sends, so it's acknowledged directly even with no other matching
  words.
- **Intent matching** — every example phrase becomes two TF-IDF vectors: a
  *word* signal (content words, accent-folded) and a *character* signal (shrugs
  off typos like "helo" → "hello"). An incoming message is compared to
  everything the bot knows; if it's confident, it replies with that intent.
  Pure pleasantries are demoted when a real question is also present, so
  "Good evening, when does my order arrive?" answers the question.
- **Fallback reasoner** — if nothing matches well, the bot inspects the message
  (question? complaint? small talk? gibberish?) and crafts a reasonable,
  language-matched reply instead of going silent.

**Multi-turn flows** — the brain reports what it expects next (e.g. a tracking
number after a tracking question); `responder.py` feeds that back so the bot
runs short conversations offline too.

**Self-improving** — unrecognised messages are logged to
`unknown_messages.log` (with detected language + confidence), and you can call
`brain.learn(tag, phrase, response, lang)` to add knowledge at runtime (it
retrains instantly).

## Tests

```powershell
python test_brain.py            # quick, no extra deps
# or: python -m pytest test_brain.py -q
```

They check behaviour that matters — right intent, right language, typo
tolerance, entity handling, multi-turn follow-ups, and graceful fallback.

## Project layout

| File | Purpose |
|------|---------|
| `brain.py` | The local AI brain (language + entities + matching + fallback + learning) |
| `knowledge.json` | What the bot knows, bilingual (en/sq) — **edit this to customise it** |
| `test_brain.py` | Behavioural tests for the brain |
| `app.py` | Flask webhook server Facebook talks to |
| `messenger.py` | Sends replies via the Facebook Send API |
| `config.py` | Loads settings from `.env` |
| `.env.example` | Template for your secrets — copy to `.env` |

## 1. Install

```powershell
cd "$env:USERPROFILE\Desktop\ai-bot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Try the brain offline (no Facebook needed)

```powershell
python brain.py
```

Type messages and watch it reply — including off-script ones. This is the
fastest way to tune `knowledge.json`.

## 3. Connect it to Facebook Messenger

You need a **Facebook Page** and a **Meta developer app**.

1. Go to <https://developers.facebook.com/> → **My Apps** → **Create App** →
   choose **Business** → add the **Messenger** product.
2. Under **Messenger → Settings**, link your Facebook Page and click
   **Generate token** → that's your `PAGE_ACCESS_TOKEN`.
3. Under **App Settings → Basic**, copy the **App Secret** → `APP_SECRET`.
4. Copy `.env.example` to `.env` and fill in the values. Make up any random
   string for `VERIFY_TOKEN` (you'll reuse it in the next step).

   ```powershell
   Copy-Item .env.example .env
   ```

5. Facebook must reach your server over **public HTTPS**. For local testing,
   run the app and expose it with a tunnel:

   ```powershell
   python app.py                # terminal 1 (runs on port 5000)
   ngrok http 5000              # terminal 2 -> gives an https URL
   ```

   (Install ngrok from <https://ngrok.com/>, or deploy to Render/Railway/Fly.)

6. Back in **Messenger → Settings → Webhooks → Add Callback URL**:
   - **Callback URL**: `https://<your-ngrok-url>/webhook`
   - **Verify Token**: the same `VERIFY_TOKEN` you put in `.env`
   - Subscribe to the **`messages`** field.

7. Message your Page from Facebook — the bot replies. 🎉

> While your app is in **Development** mode, only you and people with a role on
> the app can message it. To open it to everyone, submit the
> `pages_messaging` permission for **App Review**.

## Customising

- Add your real Q&A to `knowledge.json`. Each intent has `patterns` and
  `responses` per language (`en`, `sq`) — add phrases in either, and add new
  languages by adding a key. The bot replies in the customer's detected
  language and falls back to English if an intent lacks one.
- Tune `CONFIDENCE_THRESHOLD` in `brain.py` (lower = looser matching).
- Edit the language-aware fallback/system replies in `brain.py`
  (`SYSTEM_MESSAGES`) to match your bot's voice, and the `SQ_MARKERS` /
  `EN_MARKERS` sets if you want to tune language detection.
- Run `python test_brain.py` after changes to confirm nothing regressed.

## Hosting it live for free (Render)

Running locally needs your PC + a tunnel. To run 24/7, deploy to
[Render](https://render.com) (free, no card):

1. Push this repo to GitHub.
2. On Render: **New + → Blueprint** → pick the repo (it reads `render.yaml`).
3. When prompted, paste the secret env vars: `VERIFY_TOKEN`,
   `PAGE_ACCESS_TOKEN`, `APP_SECRET`, `OPENAI_API_KEY`.
4. Deploy. Render gives you a permanent URL like
   `https://ai-bot-xxxx.onrender.com`.
5. In Meta → Messenger → Webhooks, set the callback URL to
   `https://ai-bot-xxxx.onrender.com/webhook` (same verify token).

Notes:
- The free tier **sleeps after ~15 min idle**, so the first message after a
  quiet spell is delayed ~30–60s. Keep it awake with a free pinger like
  [cron-job.org](https://cron-job.org) hitting your `/` URL every 10 min.
- The filesystem is ephemeral, so runtime learning (`knowledge.json` edits,
  `unknown_messages.log`) resets on redeploy. The bot itself works fine.

## Security

`.env` and `*.log` are git-ignored — never commit your tokens. `messenger.py`
verifies Facebook's request signature with your `APP_SECRET` in production.

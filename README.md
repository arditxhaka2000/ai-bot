# ai-bot

A Facebook/Instagram **Messenger** bot for **Cargoteer**, a truck-dispatching
service for owner-operators and small fleets. It answers carriers as a **human
dispatch representative** (never identifies as a bot), handles pricing the way
the team wants, and captures leads. An LLM (Gemini) writes the replies; a free,
self-contained **local brain** is the always-on fallback so the bot is never
down. Trilingual (English, Spanish, Russian).

## How the brain works

Four cooperating layers (see `brain.py`):

- **Language detection** — detects **English**, **Spanish**, or **Russian**
  (Cyrillic script for Russian; Spanish characters/marker words otherwise) and
  replies in that language. Language is *sticky* across a conversation.
- **Intent matching** — every example phrase becomes two TF-IDF vectors: a
  *word* signal (content words, accent-folded) and a *character* signal (shrugs
  off typos like "helo" → "hello"). An incoming message is compared to
  everything the bot knows; if it's confident, it replies with that intent.
  Pure pleasantries are demoted when a real question is also present, so
  "Hey, what do you charge?" answers the pricing question.
- **Fallback reasoner** — if nothing matches well, the bot inspects the message
  (question? complaint? small talk? gibberish?) and crafts a reasonable,
  language-matched reply instead of going silent.

**Self-improving** — unrecognised messages are logged to
`unknown_messages.log` (with detected language + confidence), and you can call
`brain.learn(tag, phrase, response, lang)` to add knowledge at runtime (it
retrains instantly).

## The smart (LLM) brain — answers anything, on-brand

The local brain above is fast, free, and never down — but it only answers what's
in `knowledge.json`. To let the bot **formulate brand-new sentences for
questions you never pre-wrote** (in English, Spanish, or Russian), add an LLM key and it
becomes the *first* brain; the local brain stays as the always-on fallback.

- **Provider** — auto-selected: **Gemini** if `GEMINI_API_KEY` is set, else
  **OpenAI** if `OPENAI_API_KEY` is set, else local-only. Force one with
  `LLM_PROVIDER=gemini|openai|local`.
- **Free option (recommended): Google Gemini.** Get a free key (no card) at
  <https://aistudio.google.com/app/apikey>, put it in `.env` as
  `GEMINI_API_KEY=...`, done.
- **Grounded in your facts.** The LLM's system prompt is built from
  `config.SYSTEM_PROMPT` *plus* the company facts pulled from `knowledge.json`,
  so it uses the real Cargoteer services/answers instead of inventing them — one
  source of truth. Edit `knowledge.json` and both brains update.
- **Human dispatch persona + pricing rules.** It speaks as a real dispatch
  teammate (never a bot). On pricing it stays competitive, asks the carrier's
  current rate plus equipment/lanes, offers a customized quote from the team,
  and **never promises a specific rate or discount**. Off-topic questions are
  politely steered back (see `config.SYSTEM_PROMPT`).
- **Never down.** If the LLM is unavailable or errors, the bot silently falls
  back to the local brain.

Quick check after adding a key:

```powershell
python try_llm.py
```

## Production features

Beyond the two brains, the bot includes what a real customer-facing service
needs:

- **Human handoff** — asking for "an agent"/"a real person" hands the chat to
  your team: the bot goes silent so a teammate can take over, until the carrier
  types "continue" to resume.
- **In-order replies** — messages from the same person are serialized with a
  per-sender lock, so a quick second message can't get answered before the
  first (fixes the out-of-order race).
- **Rich Messenger UX** — typing indicators, "seen" receipts, quick-reply
  buttons, and a one-time `setup_messenger_profile.py` that adds a Get Started
  button, greeting, ice-breakers, and a persistent menu.
- **Robust webhook** — duplicate-message protection (Meta re-delivers on
  timeout), long-message splitting, send retries, and handling of attachments
  (authority/MC, insurance, rate cons), postbacks, and quick replies. Also
  accepts **Instagram** events (same code path).
- **Persistent memory** — conversation history, per-user state (language,
  awaiting slot, handoff), an audit trail, and captured leads (`store.py`), so
  the bot remembers across restarts. The LLM gets real multi-turn context
  ("what's my name?" works). **Dual backend, auto-selected:** a free cloud DB
  (**Turso/libSQL**) when `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` are set — so
  it persists on Render too — otherwise a local SQLite file. Check yours with
  `python try_db.py`.
- **Lead capture** — actionable requests (get-started/sign-up, pricing,
  competitor-rate mentions, escalations) are saved as a team worklist.
- **Auto-escalation** — repeated frustration hands the chat to a human and
  records a complaint lead.
- **Team notifications** — set `ADMIN_WEBHOOK_URL` and the bot pings your
  Slack/Discord on handoff and new leads.
- **Admin API** — with `ADMIN_TOKEN` set: `GET /admin/leads`,
  `/admin/conversations`, `/admin/handoffs` (pass `?token=...`). `GET /stats`
  is open and summarises volume by brain, open leads, and active handoffs.
- **Abuse/cost guard** — per-user rate limiting.

## Tests

```powershell
python test_brain.py            # local-brain behaviour (language, intents, …)
python test_features.py         # tracking, quotes, handoff, webhook, limits
# or: python -m pytest -q
```

They check behaviour that matters — right intent, right language, typo
tolerance, entity handling, multi-turn flows, real tracking/quote answers,
human handoff, rate limiting, and graceful fallback.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Flask webhook server Meta talks to (text, attachments, postbacks, IG) |
| `responder.py` | Orchestrator: handoff, rate limit, tracking/quote, LLM→local routing |
| `llm.py` | Generative brain — Gemini/OpenAI, grounded in your knowledge base |
| `brain.py` | The local AI brain (language + entities + matching + fallback + learning) |
| `knowledge.json` | What the bot knows, trilingual (en/es/ru) — **edit this to customise it** |
| `store.py` | Persistence (SQLite/Turso): history, state, audit, leads |
| `messenger.py` | Send API wrapper: typing, splitting, retries, quick replies, dedup |
| `setup_messenger_profile.py` | One-time Get Started / menu / ice-breakers setup |
| `config.py` | Loads settings from `.env` |
| `test_brain.py`, `test_features.py` | Test suites |
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
   `PAGE_ACCESS_TOKEN`, `APP_SECRET`, and (for the smart brain) `GEMINI_API_KEY`
   — or leave the LLM key blank to run on the free local brain only.
4. Deploy. Render gives you a permanent URL like
   `https://ai-bot-xxxx.onrender.com`.
5. In Meta → Messenger → Webhooks, set the callback URL to
   `https://ai-bot-xxxx.onrender.com/webhook` (same verify token).

Notes:
- The free tier **sleeps after ~15 min idle**, so the first message after a
  quiet spell is delayed ~30–60s. Keep it awake with a free pinger like
  [cron-job.org](https://cron-job.org) hitting your `/` URL every 10 min.
- The filesystem is ephemeral, so a *local* SQLite DB and runtime
  `knowledge.json` edits reset on redeploy. Set `TURSO_DATABASE_URL` +
  `TURSO_AUTH_TOKEN` (free cloud DB) so conversation history, leads, and state
  **persist across deploys**. The bot works fine either way.

## Security

`.env` and `*.log` are git-ignored — never commit your tokens. `messenger.py`
verifies Facebook's request signature with your `APP_SECRET` in production.

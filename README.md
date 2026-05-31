# ai-bot

A Facebook **Messenger** bot in Python with its own **self-contained brain** —
no Claude, no OpenAI, no paid API. It learns from a knowledge base and reasons
about messages it has never seen before, so it always replies sensibly even
when someone goes "out of protocol".

## How the brain works

- **Knowledge base** (`knowledge.json`) — example phrases ("intents") and the
  replies for each. Edit this file to teach the bot about your business.
- **Intent matching** — every example is turned into a TF-IDF vector
  (character n-grams, so it shrugs off typos). An incoming message is compared
  to everything the bot knows; if it's confident, it replies with that intent.
- **Fallback reasoner** — if nothing matches well, the bot inspects the message
  (is it a question? a complaint? small talk? gibberish?) and crafts a
  reasonable reply on its own instead of going silent.
- **Self-improving** — unrecognised messages are logged to
  `unknown_messages.log`, and you can call `brain.learn(tag, phrase, response)`
  to add new knowledge at runtime (it retrains instantly).

## Project layout

| File | Purpose |
|------|---------|
| `brain.py` | The local AI brain (matching + fallback + learning) |
| `knowledge.json` | What the bot knows — **edit this to customise it** |
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

- Add your real Q&A to `knowledge.json` (greetings, pricing, hours, FAQs…).
- Tune `CONFIDENCE_THRESHOLD` in `brain.py` (lower = looser matching).
- Edit the fallback replies in `brain.py` to match your bot's voice.

## Security

`.env` and `*.log` are git-ignored — never commit your tokens. `messenger.py`
verifies Facebook's request signature with your `APP_SECRET` in production.

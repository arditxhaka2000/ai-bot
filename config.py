"""Loads settings from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
APP_SECRET = os.getenv("APP_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))

# --- OpenAI (optional). If OPENAI_API_KEY is blank, the bot uses the free
# local brain instead. ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a friendly, helpful assistant replying to people on Facebook "
    "Messenger. Always reply in the SAME language the user wrote in "
    "(for example Albanian, English, etc.). Keep replies short, clear, and "
    "warm. If you don't know something, say so honestly and offer to connect "
    "them with a human.",
)
# How many previous turns to remember per user (keeps context without
# sending the whole history every time).
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))

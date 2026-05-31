"""Loads settings from the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
APP_SECRET = os.getenv("APP_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))

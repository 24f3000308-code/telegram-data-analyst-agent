"""
config.py — single source of truth for environment/config.
Fails fast and loudly if a required secret is missing, instead of the bot
silently crashing on the first Telegram message.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"[config] FATAL: required env var {name} is not set. "
              f"Check your .env / deployment secrets.", file=sys.stderr)
        sys.exit(1)
    return val


# --- required ---
TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = _require("OPENAI_API_KEY")

# --- public URL of THIS deployment, used to build log_url ---
# e.g. https://my-agent.onrender.com  (no trailing slash)
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8080").rstrip("/")

# --- model / agent behaviour ---
OPENAI_MODEL = os.environ.get(
    "OPENAI_MODEL",
    "gpt-5-mini"
)
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "14"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TOOL_TIMEOUT_SECONDS = int(os.environ.get("TOOL_TIMEOUT_SECONDS", "45"))
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "10"))

# --- API retry behaviour (Anthropic 429/529/timeouts) ---
API_MAX_RETRIES = int(os.environ.get("API_MAX_RETRIES", "5"))
API_BACKOFF_BASE_SECONDS = float(os.environ.get("API_BACKOFF_BASE_SECONDS", "2.0"))

# --- networking for the log server thread ---
PORT = int(os.environ.get("PORT", "8080"))

# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "run.jsonl")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LOG_URL = f"{PUBLIC_BASE_URL}/run.jsonl"

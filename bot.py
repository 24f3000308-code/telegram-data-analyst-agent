"""
bot.py — Telegram front end.

Runs two things in one process:
  1. A background thread serving logs/run.jsonl over plain HTTP, so
     config.LOG_URL (<PUBLIC_BASE_URL>/run.jsonl) is publicly wget-able.
  2. The Telegram bot itself via long polling (asyncio, python-telegram-bot
     v21) — no inbound webhook/public port needed for Telegram itself, only
     outbound, which keeps deployment simple.

Per-chat conversation history is kept in memory (bounded, see
config.MAX_HISTORY_MESSAGES) to support the multi-turn "answer the last
message" requirement without unbounded memory/token growth.
"""

import asyncio
import http.server
import json
import logging
import os
import socketserver
import threading

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

import config
from agent import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

CONVERSATIONS: dict[int, list[str]] = {}
_conv_lock = threading.Lock()


# --------------------------------------------------------------------------
# Log file HTTP server (background thread, no extra dependency needed)
# --------------------------------------------------------------------------

def _start_log_server():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=config.LOG_DIR, **kwargs)

        def log_message(self, fmt, *args):
            log.info("log-server: " + fmt, *args)

    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingServer(("0.0.0.0", config.PORT), Handler)
    log.info(f"Log server serving {config.LOG_DIR} on 0.0.0.0:{config.PORT} "
             f"(-> {config.LOG_URL})")
    server.serve_forever()


# --------------------------------------------------------------------------
# Telegram handlers
# --------------------------------------------------------------------------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    if not text:
        return

    with _conv_lock:
        history = CONVERSATIONS.setdefault(chat_id, [])
        history.append(text)
        history[:] = history[-config.MAX_HISTORY_MESSAGES:]
        snapshot = list(history)

    await update.message.chat.send_action("typing")

    loop = asyncio.get_running_loop()
    try:
        answer = await loop.run_in_executor(None, answer_question, snapshot, chat_id)
    except Exception:  # noqa: BLE001 — never let an unhandled error skip the reply
        log.exception("agent crashed for chat_id=%s", chat_id)
        answer = {"answer": None, "log_url": config.LOG_URL}

    reply_text = json.dumps(answer, ensure_ascii=False)
    await update.message.reply_text(reply_text)


async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error", exc_info=context.error)


def main():
    threading.Thread(target=_start_log_server, daemon=True).start()

    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.add_error_handler(on_error)

    log.info("Starting Telegram polling…")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

"""
agents.py — glues everything together.

Responsibilities:
1. Agent loop: send the conversation + TOOLS to the OpenAI-compatible chat
   endpoint, execute whatever tool calls come back via tools.DISPATCH, feed
   results back in, repeat until the model returns a plain text reply (no
   tool calls) or MAX_AGENT_STEPS is hit.
2. Telegram wiring: one message in -> one JSON reply out. Keeps a small
   per-chat history (MAX_HISTORY_MESSAGES) so multi-turn questions work,
   but the final reply is graded on the LAST user message only, per spec.
3. Log server: a bare stdlib HTTP server that serves LOG_FILE at /run.jsonl
   on PORT, so config.LOG_URL is actually fetchable by the grader.

Everything here fails loud into the log rather than silently — if the
agent errors out, we still try to send back *some* JSON so the bot doesn't
just go quiet on the grading account.
"""

import http.server
import json
import re
import socketserver
import sys
import threading
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

import config
import logger
import tools

client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = f"""You are a data-analyst agent operating over Telegram.

You will be given a data-analysis question. It may embed data inline, or
point at a public dataset (MOSPI, data.gov.in, RBI, etc). You have three
tools:

- web_search(query): find the current URL of a dataset/resource page.
  Dataset portals reorganize constantly — search rather than guess a URL
  from memory.
- fetch_url(url): download a file to local disk (cached) and get back its
  local path, so you can pd.read_csv/pd.read_excel it in run_python.
- run_python(code): a sandboxed subprocess with pandas/numpy/requests/bs4
  preloaded. Use this for all real computation — never compute numeric
  answers in your head, always verify with code.

Work step by step: locate the data, fetch it, load/clean it in Python,
compute the actual answer, and only then respond.

The user's question will specify the exact JSON shape it wants for the
"answer" field, and will ask you to reply with ONLY that JSON object.
When you give your FINAL reply (no more tool calls needed):
- Reply with ONLY a single JSON object, nothing else — no markdown fences,
  no commentary before or after.
- Match the "answer" shape exactly as the question specifies.
- Always include "log_url": "{config.LOG_URL}" as the second key, exactly
  as given here.

If you cannot find or compute a confident answer after reasonable effort,
still return your best-effort JSON in the requested shape rather than
prose — an approximate structured answer is worth more than an explanation.
"""

# --------------------------------------------------------------------------
# Per-chat history (kept small; multi-turn questions only need recent context)
# --------------------------------------------------------------------------

_histories: dict[int, list[dict]] = {}
_hist_lock = threading.Lock()


def _get_history(chat_id: int) -> list[dict]:
    with _hist_lock:
        return list(_histories.get(chat_id, []))


def _append_history(chat_id: int, role: str, content: str):
    with _hist_lock:
        h = _histories.setdefault(chat_id, [])
        h.append({"role": role, "content": content})
        # keep only the most recent MAX_HISTORY_MESSAGES *user* turns worth
        # of context; trim from the front once we exceed a safe cap.
        cap = max(config.MAX_HISTORY_MESSAGES * 2, 4)
        if len(h) > cap:
            del h[: len(h) - cap]


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def _call_model_with_retries(messages: list[dict], log):
    """Wraps the chat completion call with exponential backoff on
    429 / 5xx / transient network errors."""
    last_err = None
    for attempt in range(config.API_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                max_tokens=config.MAX_TOKENS,
                messages=messages,
                tools=tools.TOOLS,
                tool_choice="auto",
            )
            return resp
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = config.API_BACKOFF_BASE_SECONDS * (2 ** attempt)
            log("api_retry", attempt=attempt, error=str(e), wait_seconds=wait)
            time.sleep(wait)
    log("api_exhausted", error=str(last_err))
    raise RuntimeError(f"model call failed after {config.API_MAX_RETRIES} retries: {last_err}")


def _coerce_json(text: str, log) -> str:
    """Best-effort cleanup so the reply is always valid, single-object JSON.

    The system prompt asks for bare JSON, but models occasionally wrap it in
    ```json fences or a leading/trailing sentence. This strips fences and
    extracts the outermost {...} span; if the result still doesn't parse, or
    is missing log_url, we patch/replace as needed rather than shipping
    something the grader's json.loads() would choke on.
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()

    start, end = stripped.find("{"), stripped.rfind("}")
    candidate = stripped[start:end + 1] if start != -1 and end != -1 and end > start else stripped

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        log("json_coerce_failed", raw=text[:2000])
        return json.dumps({"answer": None, "log_url": config.LOG_URL})

    if "log_url" not in obj:
        obj["log_url"] = config.LOG_URL
    return json.dumps(obj, ensure_ascii=False)


def run_agent(chat_id: int, user_text: str) -> str:
    """Runs the full tool-use loop for one incoming message and returns the
    final reply text (expected to be a single JSON object)."""
    run_id = logger.new_run_id()
    log = logger.make_run_logger(run_id, chat_id)
    log("run_start", user_text=user_text)

    _append_history(chat_id, "user", user_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_history(chat_id)

    final_text = None
    try:
        for step in range(config.MAX_AGENT_STEPS):
            resp = _call_model_with_retries(messages, log)
            choice = resp.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                raw_text = (msg.content or "").strip()
                final_text = _coerce_json(raw_text, log)
                log("run_final", step=step, raw_reply=raw_text, cleaned_reply=final_text)
                break

            # record the assistant's tool-call turn, then execute each tool
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                log("tool_call", step=step, tool=name, args=args)
                handler = tools.DISPATCH.get(name)
                if handler is None:
                    result = f"ERROR: unknown tool '{name}'"
                else:
                    try:
                        result = handler(args)
                    except Exception as e:  # noqa: BLE001
                        result = f"ERROR: tool '{name}' raised: {e}"

                log("tool_result", step=step, tool=name, result=result[:2000])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            log("run_step_limit_hit", max_steps=config.MAX_AGENT_STEPS)

    except Exception as e:  # noqa: BLE001
        log("run_error", error=str(e))

    if not final_text:
        # best-effort fallback so the bot never goes silent
        final_text = json.dumps({"answer": None, "log_url": config.LOG_URL})
        log("run_fallback", reply=final_text)

    _append_history(chat_id, "assistant", final_text)
    return final_text


# --------------------------------------------------------------------------
# Log server: serves config.LOG_FILE at /run.jsonl on config.PORT
# --------------------------------------------------------------------------

class _LogRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=config.LOG_DIR, **kwargs)

    def log_message(self, fmt, *args):
        pass  # keep stdout clean; real events go through logger.py


def _run_log_server():
    with socketserver.TCPServer(("0.0.0.0", config.PORT), _LogRequestHandler) as httpd:
        httpd.serve_forever()


# --------------------------------------------------------------------------
# Telegram wiring
# --------------------------------------------------------------------------

async def _on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text

    reply = run_agent(chat_id, user_text)
    await update.message.reply_text(reply)


def main():
    threading.Thread(target=_run_log_server, daemon=True).start()
    print(f"[agents] log server serving {config.LOG_FILE} at {config.LOG_URL}", file=sys.stderr)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_message))

    print("[agents] bot polling started", file=sys.stderr)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

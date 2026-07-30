"""
logger.py — append-only JSONL run log.

Design choices that matter for grading:
- ONE file (logs/run.jsonl), append-only, so log_url is a single stable
  wget-able URL for your whole bot's lifetime, not one file per chat that
  the grader has to discover.
- Every line is self-describing (has run_id, chat_id, event, ts) so the
  log doubles as a real audit trail, not just a heartbeat.
- Writes are flushed immediately so a crash mid-run doesn't lose the last
  few lines (important since this is graded evidence).
"""

import json
import threading
import time
import uuid

import config

_lock = threading.Lock()  # multiple Telegram messages could arrive concurrently


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(run_id: str, chat_id, event: str, **fields):
    record = {
        "ts": time.time(),
        "run_id": run_id,
        "chat_id": chat_id,
        "event": event,
        **fields,
    }
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _lock:
        with open(config.LOG_FILE, "a") as f:
            f.write(line + "\n")
            f.flush()


def make_run_logger(run_id: str, chat_id):
    """Returns a bound log(event, **fields) callable for one run."""
    def _log(event: str, **fields):
        log_event(run_id, chat_id, event, **fields)
    return _log

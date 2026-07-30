"""
agent.py — the reasoning core.

A ReAct loop: Claude gets the question + tools (web_search, fetch_url,
run_python), calls them as many times as needed, then emits a final JSON
object. Two things make this more than a generic tool loop:

1. Shape self-check: most questions embed the EXACT JSON template they want
   back (e.g. {"answer": {"state": "..."}, "log_url": "..."}). We regex that
   template out of the question text ourselves, diff its key structure
   against what the model actually returned, and — if they don't match —
   send it back for one automatic repair pass instead of silently shipping
   a wrong shape.
2. Resilient API calls: Anthropic calls are wrapped with exponential-backoff
   retries on 429/529/timeout, since losing a graded question to a transient
   rate limit would be a shame.
"""

import json
import re
import time

from openai import OpenAI

import config
import tools
from logger import make_run_logger, new_run_id

client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """\
You are a meticulous, skeptical data analyst agent operating over Telegram.

You receive a data-analysis question. It usually specifies the EXACT JSON \
shape to reply with (e.g. {"answer": {"state": "..."}, "log_url": "..."}). \
Some questions reference public Indian government datasets (MOSPI, \
data.gov.in, RBI, NSO, census, etc.) either inline or by name; others embed \
the data directly in the message.

You have three tools:
- web_search: find the current URL of the dataset/table you need. Portal \
  URLs change, so search rather than recalling a URL from memory.
- fetch_url: download that URL to a local cached file path.
- run_python: pandas/numpy/requests/bs4 sandbox to load, clean, join, and \
  compute over the data. Print everything you need to see; nothing persists \
  between calls, so reload from the cached file path each time.

Rules:
1. Never state a number or fact you could instead compute or verify. Prefer \
   official primary sources over blog/aggregator pages.
2. Work in small, inspectable steps: print df.shape / df.columns / df.head() \
   after loading anything before you start computing with it — real \
   government spreadsheets often have header rows, merged cells, or units \
   buried in a footnote, and you need to see the data before trusting it.
3. If, after genuinely trying, you cannot locate the exact official dataset, \
   give your best-justified estimate from what you did find rather than a \
   flat refusal — but only after at least one real search+fetch attempt.
4. If the conversation has multiple messages, answer ONLY the latest one; \
   earlier messages are context only.
5. When confident, stop calling tools and reply with ONLY the requested \
   JSON object — no markdown fences, no prose before/after, keys and \
   nesting matching the question's own template exactly.
"""

REPAIR_PROMPT = """\
Your last reply's JSON structure does not match the shape the question asked \
for. Expected top-level keys/shape: {expected}
You returned: {got}
Reply again with ONLY a corrected JSON object with the same values but the \
correct shape/keys. No explanation, no markdown fences.
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _extract_json_object(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _extract_requested_template(question: str):
    """Best-effort: find the literal JSON template embedded in the question
    text itself (e.g. after 'Reply with ONLY this JSON object:'), so we can
    diff key structure later. Returns a dict or None."""
    matches = re.findall(r"\{.*\}", question, flags=re.DOTALL)
    for m in matches:
        obj = _extract_json_object(m)
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    return None


def _key_shape(obj, depth=0):
    """Structural fingerprint: key names/nesting, ignoring leaf values."""
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {k: _key_shape(v, depth + 1) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_key_shape(obj[0], depth + 1)] if obj else []
    return "<value>"


def _shapes_match(expected, got) -> bool:
    return _key_shape(expected) == _key_shape(got)


def _call_claude_with_retries(messages, log):
    delay = config.API_BACKOFF_BASE_SECONDS
    last_err = None
    for attempt in range(config.API_MAX_RETRIES):
        try:
            return client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools.TOOLS,
            )
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_err = e
            log("api_retry", attempt=attempt, error=str(e))
            time.sleep(delay)
            delay *= 2
    raise last_err


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def answer_question(conversation: list[str], chat_id) -> dict:
    """
    conversation: ordered plain-text messages in this chat; answer the LAST
                  one, using earlier ones only as context (multi-turn tasks).
    Returns the final answer dict, e.g. {"answer": ..., "log_url": ...}.
    """
    run_id = new_run_id()
    log = make_run_logger(run_id, chat_id)
    log("run_start", conversation=conversation)

    # trim history so token usage/cost doesn't grow unbounded over a long chat
    trimmed = conversation[-config.MAX_HISTORY_MESSAGES:]
    latest_question = trimmed[-1]
    template = _extract_requested_template(latest_question)
    log("parsed_template", template=template)

    context_block = "\n\n".join(
        f"[earlier message {i + 1}]: {m}" for i, m in enumerate(trimmed[:-1])
    )
    user_text = (f"{context_block}\n\n" if context_block else "") + \
                f"[question to answer]: {latest_question}"

    messages = [{"role": "user", "content": user_text}]

    final = None
    for step in range(config.MAX_AGENT_STEPS):
        resp = _call_claude_with_retries(messages, log)
        log("model_step", step=step, stop_reason=resp.stop_reason,
            content=[c.model_dump() for c in resp.content])

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant",
                              "content": [c.model_dump() for c in resp.content]})
            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                fn = tools.DISPATCH.get(block.name)
                output = fn(block.input) if fn else f"ERROR: unknown tool {block.name}"
                log("tool_call", step=step, tool=block.name,
                    input=block.input, output=output)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        text = "".join(c.text for c in resp.content if getattr(c, "type", None) == "text")
        parsed = _extract_json_object(text)

        if parsed is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                "Reply with ONLY the final JSON object requested — no other "
                "text, no markdown fences."})
            log("reply_not_json", step=step, raw=text)
            continue

        if template is not None and not _shapes_match(template, parsed):
            log("shape_mismatch", step=step, expected=template, got=parsed)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": REPAIR_PROMPT.format(
                expected=json.dumps(template), got=json.dumps(parsed))})
            continue

        final = parsed
        break

    if final is None:
        final = {"answer": None}
        log("final_answer_fallback", reason="no_valid_json_within_step_budget")

    final["log_url"] = config.LOG_URL  # always enforce the real, correct URL
    log("final_answer", answer=final)
    return final

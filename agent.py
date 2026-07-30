"""
OpenAI-based agent.py template.

NOTE:
This is a scaffold intended to replace the Anthropic-specific agent.
You may need to adjust imports if your project uses different module names.
"""

from openai import OpenAI
import json
import config
import tools
import logger

client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are an expert data-analysis agent.
Use tools when needed.
Your final answer MUST be exactly one JSON object with keys:
answer, log_url.
"""


def _call_model(messages):
    return client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=0,
        max_completion_tokens=config.MAX_TOKENS,
    )


def answer_question(question, history=None):
    history = history or []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for h in history[-config.MAX_HISTORY_MESSAGES:]:
        messages.append({"role": "user", "content": h})

    messages.append({"role": "user", "content": question})

    for step in range(MAX_AGENT_STEPS):

    response = client.chat.completions.create(
        ...
        tools=tools.TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:

        execute tools

        append tool outputs

        continue

    else:

        parse JSON

        break
    text = response.choices[0].message.content

    try:
        result = json.loads(text)
    except Exception:
        result = {
            "answer": text,
            "log_url": f"{config.PUBLIC_BASE_URL}/run.jsonl"
        }

    logger.log_run(question=question, answer=result)

    return result

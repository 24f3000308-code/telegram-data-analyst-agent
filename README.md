# Telegram Data Analyst Agent

An LLM-powered Telegram bot that answers data-analysis questions using public datasets and replies with a single JSON object in the required evaluation format.

## Features

- Telegram Bot interface
- Supports multi-turn conversations
- Uses an LLM for reasoning
- Downloads and analyzes public datasets (MOSPI and similar)
- Performs analysis using Pandas
- Produces structured JSON responses
- Generates a JSONL execution log for every request
- Deployable on Railway/Render

---

## Project Structure

```
telegram-data-analyst-agent/
│
├── bot.py                 # Telegram bot entry point
├── agent.py               # Main agent logic
├── tools.py               # Dataset utilities
├── logger.py              # JSONL logger
├── config.py
├── requirements.txt
├── README.md
│
├── prompts/
│   └── system.txt
│
├── logs/
│
├── .env.example
└── .gitignore
```

---

## Requirements

- Python 3.11+
- Telegram Bot Token
- OpenAI API Key

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/telegram-data-analyst-agent.git
cd telegram-data-analyst-agent
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
TELEGRAM_BOT_TOKEN=your_bot_token
OPENAI_API_KEY=your_openai_key
LOG_BASE_URL=https://your-domain/logs
```

---

## Running Locally

```bash
python bot.py
```

The bot will begin polling Telegram for incoming messages.

---

## Response Format

Every reply is exactly one JSON object.

Example:

```json
{
  "answer": {
    "state": "Assam"
  },
  "log_url": "https://example.com/logs/run.jsonl"
}
```

---

## Logging

Each request generates a JSONL log containing:

- User messages
- Conversation history
- Tool invocations
- Intermediate reasoning (if recorded)
- Final answer

One JSON object is written per line.

---

## Deployment

This project can be deployed on:

- Railway
- Render
- Fly.io
- Azure App Service
- Google Cloud Run

Ensure that:

- the bot remains online,
- the generated JSONL logs are publicly accessible,
- the Telegram webhook or polling process is continuously running.

---

## Testing

The evaluation pipeline referenced in the assignment can be used to test the bot locally.

Add your own questions to the provided evaluation suite and verify that the bot:

- answers correctly,
- preserves conversation context,
- returns valid JSON,
- exposes a downloadable JSONL log.

---

## License

MIT License

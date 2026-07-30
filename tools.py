"""
tools.py — the agent's hands.

Three tools, each earning its place:

1. run_python   — sandboxed subprocess (own process, wall-clock timeout,
                  CPU/memory/file-size ulimits) with pandas/requests/bs4
                  preloaded. This is how the agent actually downloads and
                  crunches MOSPI/data.gov.in tables.
2. web_search   — no-API-key DuckDuckGo HTML search, so the agent can locate
                  the *current* URL of a dataset instead of the model
                  guessing/hallucinating one from training data. Dataset
                  portals reorganize constantly; search-then-fetch is far
                  more reliable than a hardcoded link.
3. fetch_url    — downloads a URL with a disk cache (keyed by URL hash), so
                  re-asked or multi-step questions against the same table
                  don't re-download it, and so flaky/slow gov sites don't
                  tank every retry.

All three are wrapped so a failure returns a descriptive error string to the
model instead of raising — the agent should see "this failed, try another
approach" rather than the whole run crashing.
"""

import hashlib
import os
import resource
import subprocess
import tempfile
import textwrap

import requests
from bs4 import BeautifulSoup

import config

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) TelegramDataAnalystAgent/1.0 "
    "(+https://github.com/; contact: bot-operator)"
)


# --------------------------------------------------------------------------
# 1. run_python
# --------------------------------------------------------------------------

_PRELUDE = textwrap.dedent(
    """
    import pandas as pd, numpy as np, requests, json, re, io, sys, os
    from bs4 import BeautifulSoup
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    CACHE_DIR = %r
    """
) % config.CACHE_DIR


def _limit_resources():
    """Runs in the child process (preexec_fn) before exec: caps CPU time,
    address space, and open files so a runaway/adversarial snippet can't
    take down the host."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))               # 30s CPU
        resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))  # ~1.5GB
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (200_000_000, 200_000_000))   # 200MB files
    except Exception:
        pass  # not all limits are settable on every OS (e.g. some managed PaaS)


def run_python(code: str) -> str:
    full_code = _PRELUDE + "\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        path = f.name
    try:
        proc = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=config.TOOL_TIMEOUT_SECONDS,
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )
        out = proc.stdout[-6000:]
        err = proc.stderr[-3000:]
        result = f"STDOUT:\n{out or '(empty)'}"
        if err:
            result += f"\nSTDERR:\n{err}"
        if proc.returncode != 0 and not err:
            result += f"\n(exit code {proc.returncode})"
        return result
    except subprocess.TimeoutExpired:
        return (f"ERROR: execution timed out after {config.TOOL_TIMEOUT_SECONDS}s. "
                f"Break the work into smaller steps (e.g. download in one call, "
                f"analyze in the next) and cache intermediate results to CACHE_DIR.")
    except Exception as e:  # noqa: BLE001
        return f"ERROR: sandbox failed to run code: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# 2. web_search
# --------------------------------------------------------------------------

def web_search(query: str, max_results: int = 6) -> str:
    """Free, no-key DuckDuckGo HTML search. Returns 'title -- url -- snippet'
    lines. Good enough to locate MOSPI/data.gov.in/RBI resource pages, which
    is the actual bottleneck (not the search ranking quality)."""
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for res in soup.select("div.result")[:max_results]:
            link = res.select_one("a.result__a")
            snippet = res.select_one("a.result__snippet, div.result__snippet")
            if not link:
                continue
            title = link.get_text(strip=True)
            url = link.get("href", "")
            snip = snippet.get_text(strip=True) if snippet else ""
            results.append(f"- {title}\n  {url}\n  {snip}")
        if not results:
            return "No results found. Try a more specific or differently worded query."
        return "\n".join(results)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: web_search failed: {e}"


# --------------------------------------------------------------------------
# 3. fetch_url (with disk cache)
# --------------------------------------------------------------------------

def fetch_url(url: str, max_bytes: int = 20_000_000) -> str:
    """Downloads url to a cached file under config.CACHE_DIR and returns the
    local path + basic metadata, so run_python calls can pd.read_csv/
    pd.read_excel it directly without redownloading each step."""
    try:
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
        local_path = os.path.join(config.CACHE_DIR, key + ext)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return (f"CACHED at {local_path} "
                    f"({os.path.getsize(local_path)} bytes) — reuse this path, "
                    f"do not re-download.")

        with requests.get(
            url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=30
        ) as r:
            r.raise_for_status()
            size = 0
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"file exceeds {max_bytes} byte cap")
                    f.write(chunk)
        return f"Downloaded to {local_path} ({size} bytes)."
    except Exception as e:  # noqa: BLE001
        return f"ERROR: fetch_url failed for {url}: {e}"


# --------------------------------------------------------------------------
# Tool schemas handed to Claude
# --------------------------------------------------------------------------

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo) to locate the current URL of a "
            "public dataset (MOSPI, data.gov.in, RBI, NSO, census, etc.) or "
            "to check facts. Returns up to max_results 'title / url / "
            "snippet' entries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Download a URL (CSV/XLS/XLSX/HTML/JSON/PDF) to a local cache "
            "and return its local file path. Reuses the cache on repeat "
            "calls with the same URL. Use the returned path in run_python "
            "(e.g. pd.read_csv(path) or pd.read_excel(path))."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python snippet in a sandboxed subprocess (30s CPU cap, "
            "~1.5GB memory cap). Pre-imported: pandas as pd, numpy as np, "
            "requests, BeautifulSoup from bs4, json, re, io, os. A variable "
            "CACHE_DIR (string path) points at the shared download cache. "
            "Only stdout/stderr are returned — print() everything you need "
            "to see. No state persists between calls; re-load data each time "
            "or write intermediate results to files under CACHE_DIR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]

DISPATCH = {
    "web_search": lambda inp: web_search(inp.get("query", ""), inp.get("max_results", 6)),
    "fetch_url": lambda inp: fetch_url(inp.get("url", "")),
    "run_python": lambda inp: run_python(inp.get("code", "")),
}

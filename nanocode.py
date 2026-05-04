#!/usr/bin/env python3
"""nanocode - single-file OpenAI-compatible coding agent.

This file intentionally stays dependency-free and keeps related code in large,
clearly marked sections instead of splitting into modules. Reading guide:

1. Configuration/auth/constants
2. Core local tools: read/write/edit/bash plus file helpers
3. Web tools: webfetch/websearch
4. MCP discovery, inspection, and direct tool surfaces
5. Skills and context/system prompt loading
6. Headless child nanocode jobs
7. OpenAI-compatible streaming chat loop
8. Context compaction
9. Interactive slash commands, native queue/steer, and CLI entrypoint
"""

import argparse
import atexit
import concurrent.futures
import glob as globlib
import html
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# =============================================================================
# 1. Configuration, process-wide state, and auth
# =============================================================================

# Provider / model defaults ---------------------------------------------------
# All API calls go through an OpenAI-compatible /chat/completions endpoint.
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
BASE_URL = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
API_URL = os.environ.get("OPENAI_API_URL", f"{BASE_URL}/chat/completions")
MODEL = os.environ.get("MODEL", "kimi-k2.6")
PROVIDER = "OpenCode Go" if "opencode.ai/zen/go" in API_URL else "OpenAI-compatible"
SHOW_THINKING = os.environ.get("SHOW_THINKING", "1").lower() not in (
    "0",
    "false",
    "no",
)
# Harness paths and in-memory runtime state ----------------------------------
# Nanocode uses ~/.nanocode and project .nanocode files, never Pi's .pi config.
NANOCODE_DIR = Path(os.environ.get("NANOCODE_DIR", "~/.nanocode")).expanduser()
AGENT_DIR = NANOCODE_DIR
NANOCODE_SCRIPT = Path(__file__).resolve()

# Background child nanocode jobs are intentionally process-local.
NANOCODE_JOBS = {}
NANOCODE_JOB_COUNTER = 0
NANOCODE_JOB_LOCK = threading.Lock()
NANOCODE_JOB_EVENTS = queue.Queue()
NANOCODE_JOB_AUTO_EMIT_MAX_CHARS = int(
    os.environ.get("NANOCODE_JOB_AUTO_EMIT_MAX_CHARS", "18000")
)

# Native queues apply only to the parent interactive process.
NATIVE_QUEUE_LOCK = threading.RLock()
NATIVE_STEERING_QUEUE = []
NATIVE_FOLLOWUP_QUEUE = []


def env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(name, default, minimum=None, maximum=None):
    try:
        value = float(os.environ.get(name, default))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


# Compaction knobs ------------------------------------------------------------
CONTEXT_WINDOW_TOKENS = env_int(
    "NANOCODE_CONTEXT_WINDOW_TOKENS", 128_000, 4_000, 2_000_000
)
COMPACTION_REMAINING_RATIO = env_float(
    "NANOCODE_COMPACTION_REMAINING_RATIO", 0.15, 0.0, 1.0
)
if "NANOCODE_COMPACTION_TRIGGER_RATIO" in os.environ:
    COMPACTION_REMAINING_RATIO = 1.0 - env_float(
        "NANOCODE_COMPACTION_TRIGGER_RATIO", 0.85, 0.0, 1.0
    )
COMPACTION_RESERVE_TOKENS = env_int(
    "NANOCODE_COMPACTION_RESERVE_TOKENS", 16_384, 1_000, 1_000_000
)
COMPACTION_KEEP_RECENT_TOKENS = env_int(
    "NANOCODE_COMPACTION_KEEP_RECENT_TOKENS", 20_000, 1_000, 1_000_000
)
COMPACTION_ENABLED = os.environ.get("NANOCODE_COMPACTION", "1").lower() not in (
    "0",
    "false",
    "no",
)
CHILD_SYSTEM_PROMPT = """You are a headless child nanocode process.
Return compact findings only: answer, evidence paths, commands run if relevant, confidence, and next action. Default to 40 lines or fewer unless the user explicitly asks for detail.
Do not start nested nanocode jobs unless the user explicitly asks for nested orchestration.
Do not poll background jobs in a loop; if you start one, report the job id and stop."""

# Terminal colours ------------------------------------------------------------
# ANSI colors used by the simple terminal renderer.
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)


# Provider auth ---------------------------------------------------------------


def load_opencode_api_key():
    if "opencode.ai/zen/go" not in API_URL:
        return "", "none"

    auth_path = Path(
        os.environ.get("OPENCODE_AUTH_JSON", "~/.local/share/opencode/auth.json")
    ).expanduser()
    try:
        auth = json.loads(auth_path.read_text())
    except Exception:
        return "", "none"

    for provider in ("opencode-go", "opencode"):
        entry = auth.get(provider, {})
        if entry.get("type") == "api" and entry.get("key"):
            return entry["key"], f"{auth_path}:{provider}"
    return "", "none"


def find_api_key():
    for name in ("OPENCODE_API_KEY", "OPENCODE_GO_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value, name
    return load_opencode_api_key()


API_KEY, API_KEY_SOURCE = find_api_key()


# Small shared helpers --------------------------------------------------------


def clamp_number(value, fallback, minimum, maximum):
    try:
        number = int(value)
    except Exception:
        return fallback
    return max(minimum, min(maximum, number))


def truncate_middle(text, max_chars, label="truncated"):
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max(0, max_chars - head - 120)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[{label} {omitted} characters from the middle]\n\n{text[len(text) - tail:]}"


def text_from_tool_result(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def http_headers(accept="application/json"):
    return {
        "Accept": accept,
        "User-Agent": os.environ.get("NANOCODE_USER_AGENT", "nanocode/1.0"),
    }


# =============================================================================
# 2. Core local tools: read / write / edit / bash and file helpers
# =============================================================================


TOOL_MAX_LINES = 2000
TOOL_MAX_CHARS = 50 * 1024


def truncate_lines_and_chars(text, max_lines=TOOL_MAX_LINES, max_chars=TOOL_MAX_CHARS, mode="head"):
    lines = text.splitlines(keepends=True)
    truncated_lines = len(lines) > max_lines
    if truncated_lines:
        lines = lines[:max_lines] if mode == "head" else lines[-max_lines:]
    text = "".join(lines)
    truncated_chars = len(text) > max_chars
    if truncated_chars:
        text = text[:max_chars] if mode == "head" else text[-max_chars:]
    if truncated_lines or truncated_chars:
        direction = "Showing first" if mode == "head" else "Showing last"
        text += f"\n[{direction} {max_lines} lines / {max_chars // 1024}KB; output truncated]"
    return text


def resolve_tool_path(args):
    return args.get("path") or args.get("file_path")


def read(args):
    path = resolve_tool_path(args)
    if not path:
        return "error: path is required"
    try:
        lines = open(path, encoding="utf-8").readlines()
    except Exception as err:
        return f"error: could not read {path}: {err}"

    # Pi-compatible offset is 1-indexed. Omitted offset starts at line 1.
    offset = clamp_number(args.get("offset"), 1, 1, 10**12)
    if lines and offset > len(lines):
        return f"error: offset {offset} is beyond end of file ({len(lines)} lines total)"
    limit = args.get("limit")
    limit = clamp_number(limit, len(lines) or 1, 1, TOOL_MAX_LINES) if limit is not None else len(lines)
    start = offset - 1
    selected = lines[start : start + limit]
    output = "".join(f"{start + idx + 1:4}| {line}" for idx, line in enumerate(selected))
    if start + limit < len(lines):
        output += f"\n[{len(lines) - (start + limit)} more lines in file. Use offset={start + limit + 1} to continue.]"
    return truncate_lines_and_chars(output, mode="head")


def write(args):
    path = resolve_tool_path(args)
    if not path:
        return "error: path is required"
    content = args.get("content")
    if not isinstance(content, str):
        return "error: content must be a string"
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as err:
        return f"error: could not write {path}: {err}"
    return f"Successfully wrote {len(content)} bytes to {path}"


def normalize_edit_args(args):
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except Exception:
            return None, "edits must be an array, not an unparsable string"
    if not edits:
        old_text = args.get("oldText", args.get("old"))
        new_text = args.get("newText", args.get("new"))
        if isinstance(old_text, str) and isinstance(new_text, str):
            edits = [{"oldText": old_text, "newText": new_text}]
    if not isinstance(edits, list) or not edits:
        return None, "edits must contain at least one replacement"
    normalized = []
    for idx, item in enumerate(edits):
        if not isinstance(item, dict) or not isinstance(item.get("oldText"), str) or not isinstance(item.get("newText"), str):
            return None, f"edits[{idx}] must include string oldText and newText"
        normalized.append({"oldText": item["oldText"], "newText": item["newText"]})
    return normalized, None


def edit(args):
    path = resolve_tool_path(args)
    if not path:
        return "error: path is required"
    edits, error = normalize_edit_args(args)
    if error:
        return f"error: {error}"
    try:
        original = open(path, encoding="utf-8").read()
    except Exception as err:
        return f"error: could not read {path}: {err}"

    spans = []
    for idx, item in enumerate(edits):
        old_text = item["oldText"]
        count = original.count(old_text)
        if count == 0:
            return f"error: edits[{idx}].oldText not found"
        if count > 1:
            return f"error: edits[{idx}].oldText appears {count} times; it must be unique"
        start = original.index(old_text)
        spans.append((start, start + len(old_text), item["newText"], idx))

    spans.sort(key=lambda span: span[0])
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            return f"error: edits[{previous[3]}] and edits[{current[3]}] overlap"

    parts = []
    cursor = 0
    for start, end, new_text, _idx in spans:
        parts.append(original[cursor:start])
        parts.append(new_text)
        cursor = end
    parts.append(original[cursor:])
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(parts))
    except Exception as err:
        return f"error: could not write {path}: {err}"
    return f"Successfully replaced {len(edits)} block(s) in {path}."


def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def ls(args):
    path = Path(args.get("path", "."))
    limit = clamp_number(args.get("limit"), 500, 1, 5000)
    try:
        entries = sorted(
            path.iterdir(),
            key=lambda entry: (not entry.is_dir(), entry.name.lower()),
        )
    except Exception as err:
        return f"error: {err}"
    lines = [entry.name + ("/" if entry.is_dir() else "") for entry in entries[:limit]]
    if len(entries) > limit:
        lines.append(f"... +{len(entries) - limit} more")
    return "\n".join(lines) or "(empty)"


def find_files(args):
    root = Path(args.get("path", "."))
    pattern = args.get("pattern") or args.get("pat") or "*"
    limit = clamp_number(args.get("limit"), 500, 1, 5000)
    include_hidden = bool(args.get("hidden"))
    try:
        matches = []
        for path in root.rglob(pattern):
            rel_parts = path.relative_to(root).parts if path != root else path.parts
            if not include_hidden and any(part.startswith(".") for part in rel_parts):
                continue
            matches.append(str(path) + ("/" if path.is_dir() else ""))
            if len(matches) >= limit:
                break
    except Exception as err:
        return f"error: {err}"
    return "\n".join(matches) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def tool_stdout_enabled():
    return os.environ.get("NANOCODE_QUIET_TOOLS", "0").lower() in (
        "0",
        "false",
        "no",
    )


def bash(args):
    command = args.get("command") or args.get("cmd")
    if not command:
        return "error: command is required"
    timeout = args.get("timeout")
    if timeout is not None:
        try:
            timeout = max(1, min(24 * 60 * 60, int(timeout)))
        except Exception:
            timeout = None

    full_command = command
    if os.environ.get("NANOCODE_MISE_HOT_RELOAD", "1").lower() not in (
        "0",
        "false",
        "no",
    ):
        full_command = 'eval "$(mise env -s bash)"\n' + full_command

    proc = subprocess.Popen(
        full_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    line_queue = queue.Queue()

    def reader():
        try:
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    threading.Thread(target=reader, daemon=True).start()
    start = time.time()
    timed_out = False
    reader_done = False
    while not reader_done:
        if timeout is not None and time.time() - start > timeout and proc.poll() is None:
            timed_out = True
            proc.kill()
        try:
            line = line_queue.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None:
                continue
            else:
                continue
        if line is None:
            reader_done = True
            continue
        if tool_stdout_enabled():
            print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
        output_lines.append(line)

    proc.wait()
    if timed_out:
        output_lines.append(f"\n(timed out after {timeout}s)")
    output = "".join(output_lines).strip()
    if proc.returncode not in (0, None):
        output = (output + f"\n(exit code {proc.returncode})").strip()
    return truncate_lines_and_chars(output or "(empty)", mode="tail")


# =============================================================================
# 3. Web tools: webfetch and websearch
# =============================================================================
# These stay stdlib-only. Exa is used through its MCP-compatible HTTP endpoint
# when configured; DuckDuckGo HTML search is the zero-config fallback.


HTML_ENTITY_NAMED = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "ndash": "-",
    "mdash": "-",
    "hellip": "...",
}


def decode_entities(text):
    # html.unescape covers the named/numeric cases used by the Pi extensions.
    return html.unescape(text)


def strip_tags(text):
    return re.sub(r"\s+", " ", decode_entities(re.sub(r"<[^>]*>", " ", text))).strip()


def normalise_whitespace(text):
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\t", " ")
        .replace(" \n", "\n")
    )


def tidy_text(text):
    text = normalise_whitespace(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def html_to_markdown(raw):
    body = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", raw, flags=re.I)
    body = re.sub(r"<style\b[^>]*>[\s\S]*?</style>", "", body, flags=re.I)
    body = re.sub(r"<noscript\b[^>]*>[\s\S]*?</noscript>", "", body, flags=re.I)
    body = re.sub(r"<!--[\s\S]*?-->", "", body)

    def link(match):
        href, label = match.group(1), match.group(2)
        text = strip_tags(label)
        return f"[{text}]({decode_entities(href)})" if text else decode_entities(href)

    body = re.sub(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>([\s\S]*?)</a>",
        link,
        body,
        flags=re.I,
    )

    def heading(match):
        level, content = int(match.group(1)), strip_tags(match.group(2))
        return f"\n\n{'#' * level} {content}\n"

    body = re.sub(r"<h([1-6])\b[^>]*>([\s\S]*?)</h\1>", heading, body, flags=re.I)
    body = re.sub(r"<li\b[^>]*>", "\n- ", body, flags=re.I)
    body = re.sub(
        r"</(p|div|section|article|header|footer|main|aside|nav|ul|ol|li|blockquote|pre|table|tr)>",
        "\n",
        body,
        flags=re.I,
    )
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    return tidy_text(decode_entities(body))


def html_to_text(raw):
    body = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", raw, flags=re.I)
    body = re.sub(r"<style\b[^>]*>[\s\S]*?</style>", "", body, flags=re.I)
    body = re.sub(r"<!--[\s\S]*?-->", "", body)
    body = re.sub(
        r"</(p|div|section|article|header|footer|main|aside|nav|li|h[1-6]|tr)>",
        "\n",
        body,
        flags=re.I,
    )
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    return tidy_text(decode_entities(body))


def ensure_http_url(raw):
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"webfetch only supports http and https URLs, got {parsed.scheme}")
    return raw


def webfetch(args):
    url = ensure_http_url(args["url"])
    timeout = clamp_number(args.get("timeoutSeconds"), 30, 1, 120)
    max_chars = clamp_number(args.get("maxCharacters"), 30_000, 1_000, 200_000)
    requested_format = args.get("format")
    max_bytes = 5 * 1024 * 1024

    request = urllib.request.Request(
        url,
        headers=http_headers(
            "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8"
        ),
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                return f"error: Response exceeded {max_bytes} byte limit"
            content_type = response.headers.get("content-type", "")
            final_url = response.geturl()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as err:
        body = err.read(max_bytes).decode("utf-8", "replace")
        content_type = err.headers.get("content-type", "") if err.headers else ""
        is_html = bool(re.search(r"text/html|application/xhtml\+xml", content_type, re.I)) or bool(
            re.search(r"<html[\s>]", body, re.I)
        )
        text = html_to_text(body) if is_html else body
        return truncate_middle(
            f"webfetch failed with HTTP {err.code} {err.reason}\n\n{text}",
            max_chars,
            "webfetch truncated",
        )

    raw = body.decode("utf-8", "replace")
    is_html = bool(re.search(r"text/html|application/xhtml\+xml", content_type, re.I)) or bool(
        re.search(r"<html[\s>]", raw, re.I)
    )
    is_textual = bool(
        re.search(r"^text/|json|xml|yaml|csv|markdown|javascript|typescript", content_type, re.I)
    )

    if requested_format == "html":
        return truncate_middle(raw, max_chars, "webfetch truncated")
    if is_html:
        text = html_to_text(raw) if requested_format == "text" else html_to_markdown(raw)
        return truncate_middle(text, max_chars, "webfetch truncated")
    if is_textual or not content_type:
        return truncate_middle(raw, max_chars, "webfetch truncated")
    return (
        f"Fetched {final_url}, but it is non-text content "
        f"({content_type or 'unknown content type'}, {len(body)} bytes)."
    )


EXA_MCP_URL = "https://mcp.exa.ai/mcp"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"


def parse_json_or_sse(raw):
    trimmed = raw.strip()
    if trimmed.startswith("{"):
        return json.loads(trimmed)
    for line in trimmed.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload and payload != "[DONE]":
            return json.loads(payload)
    raise ValueError("response was neither JSON nor SSE data")


def extract_mcp_text(result):
    content = result.get("content") if isinstance(result, dict) else None
    if content is None and isinstance(result, dict) and isinstance(result.get("result"), dict):
        content = result["result"].get("content")
    if isinstance(content, list):
        text = "\n\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
        if text:
            return text
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return result["text"]
        if isinstance(result.get("result"), str):
            return result["result"]
        return json.dumps(result.get("result", result), indent=2, ensure_ascii=False)
    return str(result)


def call_exa_search(query, num_results, search_type, livecrawl, context_max_chars):
    api_key = os.environ.get("EXA_API_KEY")
    url = EXA_MCP_URL
    if api_key:
        url += "?" + urllib.parse.urlencode({"exaApiKey": api_key})
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "numResults": num_results,
                "type": search_type,
                "livecrawl": livecrawl,
                "contextMaxCharacters": context_max_chars,
            },
        },
    }
    headers = http_headers("application/json, text/event-stream")
    headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=25) as response:
        raw = response.read().decode("utf-8", "replace")
    parsed = parse_json_or_sse(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(parsed["error"].get("message", json.dumps(parsed["error"])))
    return extract_mcp_text(parsed.get("result", parsed) if isinstance(parsed, dict) else parsed)


def unwrap_duckduckgo_url(raw):
    decoded = decode_entities(raw)
    try:
        url = urllib.parse.urlparse(urllib.parse.urljoin("https://duckduckgo.com", decoded))
        params = urllib.parse.parse_qs(url.query)
        wrapped = params.get("uddg", [None])[0]
        return urllib.parse.unquote(wrapped) if wrapped else urllib.parse.urlunparse(url)
    except Exception:
        return decoded


def search_duckduckgo(query, num_results):
    data = urllib.parse.urlencode({"q": query}).encode()
    headers = http_headers("text/html,application/xhtml+xml")
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    headers["User-Agent"] = "Mozilla/5.0 (compatible; nanocode-websearch/1.0)"
    request = urllib.request.Request(DUCKDUCKGO_HTML_URL, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read().decode("utf-8", "replace")

    results = []
    item_re = re.compile(
        r'<div[^>]+class="[^"]*result[^"]*"[^>]*>([\s\S]*?)(?=<div[^>]+class="[^"]*result[^"]*"|</body>)',
        re.I,
    )
    for item in item_re.finditer(raw):
        if len(results) >= num_results:
            break
        block = item.group(1)
        link = re.search(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>',
            block,
            re.I,
        )
        if not link:
            continue
        snippet = re.search(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</a>',
            block,
            re.I,
        ) or re.search(
            r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</div>',
            block,
            re.I,
        )
        results.append(
            {
                "title": strip_tags(link.group(2)),
                "url": unwrap_duckduckgo_url(link.group(1)),
                "snippet": strip_tags(snippet.group(1)) if snippet else "",
            }
        )

    if not results:
        return f'No DuckDuckGo results found for "{query}".'
    return "\n\n".join(
        f"{idx}. {item['title']}\n   {item['url']}"
        + (f"\n   {item['snippet']}" if item["snippet"] else "")
        for idx, item in enumerate(results, 1)
    )


def websearch(args):
    query = args["query"]
    num_results = clamp_number(args.get("numResults"), 8, 1, 10)
    provider = args.get("provider", "auto")
    search_type = args.get("type", "auto")
    livecrawl = args.get("livecrawl", "fallback")
    context_max_chars = clamp_number(args.get("contextMaxCharacters"), 10_000, 1_000, 50_000)

    if provider in ("exa", "auto"):
        try:
            text = call_exa_search(
                query, num_results, search_type, livecrawl, context_max_chars
            )
            return truncate_middle(text, context_max_chars, "websearch truncated")
        except Exception as err:
            if provider == "exa":
                return f"Exa websearch failed: {err}"

    return search_duckduckgo(query, num_results)


# =============================================================================
# 4. MCP: config loading, stdio/remote clients, router tools, direct tool specs
# =============================================================================
# Nanocode deliberately reads only ~/.nanocode/mcp.json and project
# .nanocode/mcp.json files. It does not load Pi's .pi config.


MCP_CLIENTS = {}
MCP_INVENTORY_CACHE = {"key": None, "inventory": None}


def strip_json_comments(raw):
    raw = re.sub(r"/\*[\s\S]*?\*/", "", raw)
    raw = re.sub(r"(^|[^:])//.*$", r"\1", raw, flags=re.M)
    return re.sub(r",\s*([}\]])", r"\1", raw)


def read_json_config(path):
    return json.loads(strip_json_comments(Path(path).read_text()))


def is_plain_object(value):
    return isinstance(value, dict)


def merge_config_value(base, override):
    if override is None:
        return None
    if isinstance(override, list) or not is_plain_object(override):
        return override
    result = dict(base) if is_plain_object(base) else {}
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
            continue
        merged = merge_config_value(result.get(key), value)
        if merged is None:
            result.pop(key, None)
        else:
            result[key] = merged
    return result


def merge_configs(base, override):
    merged = merge_config_value(base or {"servers": {}}, override or {"servers": {}})
    if "servers" not in merged or not isinstance(merged["servers"], dict):
        merged["servers"] = {}
    return merged


def ancestor_dirs(cwd):
    dirs = []
    path = Path(cwd).resolve()
    while True:
        dirs.append(path)
        if path.parent == path:
            break
        path = path.parent
    return list(reversed(dirs))


def project_mcp_path(path):
    return Path(path) / ".nanocode" / "mcp.json"


def load_mcp_config(cwd=None):
    cwd = cwd or os.getcwd()
    config = {"servers": {}}
    global_path = AGENT_DIR / "mcp.json"
    if global_path.exists():
        config = merge_configs(config, read_json_config(global_path))
    for path in ancestor_dirs(cwd):
        project_path = project_mcp_path(path)
        if project_path.exists():
            config = merge_configs(config, read_json_config(project_path))
    return config


def enabled_servers(config):
    return [
        (name, server)
        for name, server in (config.get("servers") or {}).items()
        if isinstance(server, dict) and server.get("enabled") is not False
    ]


def selected_model_tools(server):
    selected = server.get("selectedTools")
    return [tool for tool in selected if isinstance(tool, str)] if isinstance(selected, list) else []


def timeout_for(server):
    return clamp_number(server.get("timeoutMs"), 30_000, 1_000, 300_000) / 1000


def resolve_env_placeholder(value):
    match = re.match(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)$", value or "")
    return os.environ.get(match.group(1)) if match else value


def resolve_env_record(record, omit_missing=True):
    resolved = {}
    for key, value in (record or {}).items():
        if not isinstance(value, str):
            continue
        next_value = resolve_env_placeholder(value)
        if next_value is None and omit_missing:
            continue
        resolved[key] = next_value or ""
    return resolved


def mcp_url(server):
    raw = server.get("url") or server.get("baseUrl")
    if not raw:
        raise RuntimeError("remote MCP server is missing url/baseUrl")
    parsed = urllib.parse.urlparse(raw)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    api_key_env = server.get("apiKeyEnv")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key and parsed.hostname == "mcp.exa.ai" and "exaApiKey" not in query:
        query["exaApiKey"] = api_key
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def mcp_headers(server):
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **resolve_env_record(server.get("headers"), True),
    }
    for header, env_name in (server.get("envHeaders") or {}).items():
        if isinstance(env_name, str) and os.environ.get(env_name):
            headers[header] = os.environ[env_name]
    api_key_env = server.get("apiKeyEnv")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key and "authorization" not in {key.lower() for key in headers}:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def remote_mcp_request(server_name, server, method, params):
    payload = {"jsonrpc": "2.0", "id": int(time.time() * 1000), "method": method, "params": params}
    request = urllib.request.Request(
        mcp_url(server), data=json.dumps(payload).encode(), headers=mcp_headers(server)
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_for(server)) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raw = err.read(500).decode("utf-8", "replace")
        raise RuntimeError(f"{server_name}: HTTP {err.code}: {raw}") from None
    parsed = parse_json_or_sse(raw)
    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(parsed["error"].get("message", json.dumps(parsed["error"])))
    return parsed.get("result", parsed) if isinstance(parsed, dict) else parsed


class StdioMcpClient:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.process = None
        self.next_id = 1
        self.pending = {}
        self.initialized = False
        self.lock = threading.Lock()

    def start(self):
        if self.process:
            return
        command = self.config.get("command")
        if not command:
            raise RuntimeError(f"{self.name}: stdio MCP server is missing command")
        env = {**os.environ, **resolve_env_record(self.config.get("env"), True)}
        self.process = subprocess.Popen(
            [command, *(self.config.get("args") or [])],
            cwd=self.config.get("cwd"),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def close(self):
        if self.process:
            self.process.kill()
        self.process = None
        self.initialized = False

    def _reader(self):
        while self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except Exception:
                continue
            message_id = message.get("id")
            if message_id in self.pending:
                self.pending.pop(message_id).put(message)

    def notify(self, method, params):
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"{self.name}: MCP stdio server is not running")
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.process.stdin.flush()

    def request(self, method, params):
        self.start()
        with self.lock:
            message_id = self.next_id
            self.next_id += 1
            result_queue = queue.Queue(maxsize=1)
            self.pending[message_id] = result_queue
            message = {"jsonrpc": "2.0", "id": message_id, "method": method, "params": params}
            if not self.process or not self.process.stdin:
                raise RuntimeError(f"{self.name}: MCP stdio server is not running")
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        try:
            response = result_queue.get(timeout=timeout_for(self.config))
        except queue.Empty:
            self.pending.pop(message_id, None)
            raise RuntimeError(f"{self.name}: {method} timed out") from None
        if response.get("error"):
            raise RuntimeError(response["error"].get("message", json.dumps(response["error"])))
        return response.get("result", response)

    def initialize(self):
        if self.initialized:
            return
        self.request(
            "initialize",
            {
                "protocolVersion": self.config.get("protocolVersion", "2025-06-18"),
                "capabilities": {},
                "clientInfo": {"name": "nanocode-mcp", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        self.initialized = True

    def list_tools(self):
        self.initialize()
        return self.request("tools/list", {})

    def call_tool(self, tool, args):
        self.initialize()
        return self.request("tools/call", {"name": tool, "arguments": args or {}})


def stdio_client_key(name, server):
    return json.dumps(
        {
            "name": name,
            "command": server.get("command"),
            "args": server.get("args") or [],
            "cwd": server.get("cwd"),
            "env": server.get("env") or {},
            "protocolVersion": server.get("protocolVersion"),
            "timeoutMs": server.get("timeoutMs"),
        },
        sort_keys=True,
    )


def client_for(name, server):
    key = stdio_client_key(name, server)
    if key not in MCP_CLIENTS:
        MCP_CLIENTS[key] = StdioMcpClient(name, server)
    return MCP_CLIENTS[key]


def close_mcp_clients():
    for client in list(MCP_CLIENTS.values()):
        client.close()
    MCP_CLIENTS.clear()


atexit.register(close_mcp_clients)


def tool_allowed(server, tool_name):
    enabled = server.get("enabledTools") or server.get("allowedTools")
    if enabled and tool_name not in enabled:
        return False
    if tool_name in (server.get("disabledTools") or []):
        return False
    return True


def normalize_mcp_tools(server_name, server, result):
    raw_tools = result.get("tools") if isinstance(result, dict) else result
    if not isinstance(raw_tools, list):
        return []
    tools = []
    for tool in raw_tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        if not tool_allowed(server, tool["name"]):
            continue
        tools.append(
            {
                "server": server_name,
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema")
                or tool.get("input_schema")
                or tool.get("parameters")
                or {},
            }
        )
    return tools


def list_server_tools(server_name, server):
    kind = server.get("type") or ("stdio" if server.get("command") else "remote")
    if kind == "stdio":
        result = client_for(server_name, server).list_tools()
    else:
        result = remote_mcp_request(server_name, server, "tools/list", {})
    return normalize_mcp_tools(server_name, server, result)


def inventory_cache_key(cwd, config):
    return f"{Path(cwd).resolve()}:{json.dumps(config.get('servers', {}), sort_keys=True)}"


def load_mcp_inventory(cwd=None, refresh=False):
    cwd = cwd or os.getcwd()
    config = load_mcp_config(cwd)
    key = inventory_cache_key(cwd, config)
    if MCP_INVENTORY_CACHE["key"] == key and MCP_INVENTORY_CACHE["inventory"] and not refresh:
        return MCP_INVENTORY_CACHE["inventory"]

    servers = enabled_servers(config)
    tools, errors = [], []

    def load_one(item):
        server_name, server = item
        try:
            return list_server_tools(server_name, server), None
        except Exception as err:
            return [], f"{server_name}: {err}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(servers)))) as pool:
        for server_tools, error in pool.map(load_one, servers):
            tools.extend(server_tools)
            if error:
                errors.append(error)

    inventory = {"tools": tools, "errors": errors, "loadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    MCP_INVENTORY_CACHE.update({"key": key, "inventory": inventory})
    return inventory


def matches_mcp_query(tool, query):
    needle = query.lower()
    return (
        needle in tool["server"].lower()
        or needle in tool["name"].lower()
        or needle in tool.get("description", "").lower()
    )


def format_mcp_inventory(inventory, query=None, limit=30):
    filtered = [tool for tool in inventory["tools"] if not query or matches_mcp_query(tool, query)]
    lines = []
    for tool in filtered[:limit]:
        desc = f" - {tool['description']}" if tool.get("description") else ""
        lines.append(f"- {tool['server']}.{tool['name']}{desc}")
    if len(filtered) > limit:
        lines.append(f"... {len(filtered) - limit} more tool(s) omitted")
    if not lines:
        lines.append(f'No MCP tools matched "{query}".' if query else "No MCP tools found.")
    if inventory["errors"]:
        lines.append("\nMCP server errors:\n" + "\n".join(f"- {error}" for error in inventory["errors"]))
    return "\n".join(lines)


def find_mcp_tool(inventory, server, tool):
    matches = [candidate for candidate in inventory["tools"] if candidate["name"] == tool and (not server or candidate["server"] == server)]
    return matches[0] if len(matches) == 1 else None


def extract_call_text(result):
    if isinstance(result, dict):
        content = result.get("content")
        if content is None and isinstance(result.get("result"), dict):
            content = result["result"].get("content")
        if isinstance(content, list):
            text = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
            if text:
                return text
        if isinstance(result.get("text"), str):
            return result["text"]
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, ensure_ascii=False)


def call_mcp_tool(cwd, server_name, tool_name, tool_args):
    config = load_mcp_config(cwd)
    server = (config.get("servers") or {}).get(server_name)
    if not server or server.get("enabled") is False:
        raise RuntimeError(f'MCP server "{server_name}" is not configured or is disabled')
    kind = server.get("type") or ("stdio" if server.get("command") else "remote")
    if kind == "stdio":
        return client_for(server_name, server).call_tool(tool_name, tool_args)
    return remote_mcp_request(server_name, server, "tools/call", {"name": tool_name, "arguments": tool_args or {}})


def mcp_search(args):
    refresh = args.get("refresh") is True
    inventory = load_mcp_inventory(os.getcwd(), refresh)
    limit = clamp_number(args.get("limit"), 30, 1, 100)
    return format_mcp_inventory(inventory, args.get("query"), limit)


def mcp_inspect(args):
    inventory = load_mcp_inventory(os.getcwd(), args.get("refresh") is True)
    tool = find_mcp_tool(inventory, args.get("server"), args["tool"])
    if not tool:
        prefix = f"{args.get('server')}." if args.get("server") else ""
        return f"MCP tool not found: {prefix}{args['tool']}\n\n{format_mcp_inventory(inventory, args['tool'], 20)}"
    return json.dumps(tool, indent=2, ensure_ascii=False)


def mcp_call(args):
    result = call_mcp_tool(os.getcwd(), args["server"], args["tool"], args.get("arguments") or {})
    return extract_call_text(result)


def sanitize_tool_name_part(value):
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not sanitized:
        return "tool"
    return f"_{sanitized}" if sanitized[0].isdigit() else sanitized


def direct_mcp_tool_name(server, tool):
    return f"mcp__{sanitize_tool_name_part(server)}__{sanitize_tool_name_part(tool)}"


def parse_direct_mcp_tool_name(name):
    if not name.startswith("mcp__"):
        return None
    parts = name.split("__", 2)
    if len(parts) != 3:
        return None
    wanted_server, wanted_tool = parts[1], parts[2]
    for server_name, server in enabled_servers(load_mcp_config(os.getcwd())):
        for tool_name in selected_model_tools(server):
            if direct_mcp_tool_name(server_name, tool_name) == name:
                return server_name, tool_name
    return wanted_server, wanted_tool


def direct_mcp_specs(cwd=None):
    specs = {}
    config = load_mcp_config(cwd or os.getcwd())
    for server_name, server in enabled_servers(config):
        for tool_name in selected_model_tools(server):
            name = direct_mcp_tool_name(server_name, tool_name)
            description = server.get("description") or f"Call MCP tool {server_name}.{tool_name}"
            specs[name] = {
                "description": f"{description}\n\nRoutes to MCP server \"{server_name}\", raw tool \"{tool_name}\".",
                "params": {"type": "object", "properties": {}, "additionalProperties": True},
                "fn": lambda params, s=server_name, t=tool_name: extract_call_text(call_mcp_tool(os.getcwd(), s, t, params or {})),
                "snippet": f"{name}: direct MCP tool for {server_name}.{tool_name}.",
                "guidelines": [
                    f"Use {name} directly for requests that match {server_name}.{tool_name}; use mcp_inspect only if arguments are unclear."
                ],
            }
    return specs


# =============================================================================
# 5. Skills: progressive-disclosure SKILL.md discovery
# =============================================================================
# Project/global nanocode skills are isolated from Pi, but shared ~/.agents/skills
# is also supported so personal skills can be reused.


SKILLS = []


def find_git_root(cwd):
    path = Path(cwd).resolve()
    while True:
        if (path / ".git").exists():
            return path
        if path.parent == path:
            return Path(cwd).resolve()
        path = path.parent


def skill_search_dirs(cwd):
    cwd = Path(cwd).resolve()
    root = find_git_root(cwd)
    dirs = []

    # nanocode resource precedence is project-first, then user/global.
    # Nearest project directories win name collisions over ancestor projects;
    # ~/.nanocode/skills wins over shared ~/.agents/skills.
    ancestors = []
    path = cwd
    while True:
        ancestors.append(path)
        if path == root or path.parent == path:
            break
        path = path.parent
    for ancestor in ancestors:
        dirs.append((ancestor / ".nanocode" / "skills", True))

    dirs.append((AGENT_DIR / "skills", True))
    dirs.append((Path.home() / ".agents" / "skills", False))
    return dirs


def discover_skill_files(root, include_root_md):
    root = Path(root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    files = []
    if include_root_md:
        files.extend(path for path in root.glob("*.md") if path.name != "README.md")
    for path in root.rglob("SKILL.md"):
        if any(part in ("node_modules", ".git") for part in path.parts):
            continue
        files.append(path)
    return files


def parse_frontmatter(text):
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[text.find("\n", end + 1) + 1 :]
    lines = raw.splitlines()
    data = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not match:
            i += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in (">", "|"):
            i += 1
            block = []
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                block.append(lines[i].strip())
                i += 1
            data[key] = "\n".join(block).strip() if value == "|" else " ".join(block).strip()
            continue
        data[key] = value.strip('"\'')
        i += 1
    return data, body


def load_skills(cwd=None):
    cwd = cwd or os.getcwd()
    skills = []
    seen_paths = set()
    seen_names = set()
    for directory, include_root_md in skill_search_dirs(cwd):
        for file_path in discover_skill_files(directory, include_root_md):
            real = str(file_path.resolve())
            if real in seen_paths:
                continue
            seen_paths.add(real)
            try:
                raw = file_path.read_text()
            except Exception:
                continue
            frontmatter, _body = parse_frontmatter(raw)
            name = frontmatter.get("name") or (file_path.parent.name if file_path.name == "SKILL.md" else file_path.stem)
            description = frontmatter.get("description", "").strip()
            if not name or not description or name in seen_names:
                continue
            seen_names.add(name)
            disabled = str(frontmatter.get("disable-model-invocation", "false")).lower() == "true"
            skills.append(
                {
                    "name": name,
                    "description": description,
                    "path": str(file_path),
                    "disabled": disabled,
                }
            )
    return skills


def format_skills_for_prompt(skills):
    visible = [skill for skill in skills if not skill.get("disabled")]
    if not visible:
        return ""
    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory.",
        "Treat this metadata as routing information; read SKILL.md before following a skill.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{html.escape(skill['name'])}</name>",
                f"    <description>{html.escape(skill['description'])}</description>",
                f"    <location>{html.escape(skill['path'])}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def skill_invocation_message(name, args, skills):
    matches = [skill for skill in skills if skill["name"] == name]
    if not matches:
        return None, f"Skill not found: {name}"
    skill = matches[0]
    try:
        content = Path(skill["path"]).read_text()
    except Exception as err:
        return None, f"Could not read {skill['path']}: {err}"
    message = f"Use this skill for the next task.\n\nSkill: {skill['name']}\nPath: {skill['path']}\n\n{content}"
    if args.strip():
        message += f"\n\nUser: {args.strip()}"
    return message, None


# =============================================================================
# 6. Headless child nanocode jobs and self-call support
# =============================================================================
# The nanocode tool launches this same script in --print/--mode json form.
# Synchronous runs return directly; explicit background jobs are tracked in the
# parent process and auto-emit results at safe interactive boundaries.


def parse_tool_names(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, list):
        names = value
    else:
        raw = str(value).strip()
        lowered = raw.lower()
        if lowered in ("none", "no", "false"):
            return []
        # There is one unified nanocode tool surface. Historical preset names are
        # accepted as aliases for that unified set instead of hiding tools.
        if lowered in ("default", "all", "*", "core", "pi", "readonly", "extended", "rich", "safe"):
            return None
        names = raw.split(",")
    return [name.strip() for name in names if str(name).strip()]


def nanocode_command(prompt, tools=None, append_system_prompt="", model=None):
    tool_names = parse_tool_names(tools, None)
    command = [
        sys.executable,
        str(NANOCODE_SCRIPT),
        "--print",
        "--mode",
        "json",
        "--quiet",
        "--no-session",
    ]
    if tool_names is not None:
        command.extend(["--tools", ",".join(tool_names) if tool_names else "none"])
    if model:
        command.extend(["--model", str(model)])

    child_prompt = CHILD_SYSTEM_PROMPT
    if append_system_prompt:
        child_prompt += "\n\n" + str(append_system_prompt)

    temp_path = None
    if child_prompt:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".md") as tmp:
            tmp.write(child_prompt)
            temp_path = tmp.name
        command.extend(["--append-system-prompt", temp_path])
    command.append(prompt)
    return command, temp_path


def nanocode_child_env(model=None):
    env = os.environ.copy()
    env["NANOCODE_CHILD"] = "1"
    env["NANOCODE_QUIET_TOOLS"] = "1"
    if model:
        env["MODEL"] = str(model)
    return env


def parse_nanocode_json_output(stdout, stderr=""):
    text = ""
    data = None
    text_deltas = []
    progress = []
    last_reasoning = ""
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        event_type = event.get("type")
        if event_type == "agent_end":
            data = event
            text = event.get("text", text)
        elif event_type == "text_delta":
            text_deltas.append(event.get("delta", ""))
        elif event_type == "reasoning_delta":
            last_reasoning = (last_reasoning + event.get("delta", ""))[-500:]
        elif event_type == "tool_start":
            args_preview = json.dumps(event.get("args", {}), ensure_ascii=False)[:180]
            progress.append(f"→ {event.get('name')}({args_preview})")
        elif event_type == "tool_end":
            result = str(event.get("result", "")).split("\n", 1)[0][:180]
            progress.append(f"✓ {event.get('name')}: {result}")
        elif event_type == "message_end" and event.get("text"):
            text = event.get("text", text)
    if not text and text_deltas:
        text = "".join(text_deltas)
    if data is None and not text:
        if progress:
            text = "\n".join(progress[-20:])
        elif last_reasoning.strip():
            text = f"(thinking...) {last_reasoning.strip()}"
        else:
            text = (stdout or stderr or "").strip()
    return text or "(no output yet)", data


def run_nanocode_process(prompt, tools=None, append_system_prompt="", cwd=None, model=None, timeout_seconds=120):
    timeout_seconds = clamp_number(timeout_seconds, 120, 1, 600)
    command, temp_path = nanocode_command(prompt, tools, append_system_prompt, model)
    try:
        proc = subprocess.run(
            command,
            cwd=cwd or os.getcwd(),
            env=nanocode_child_env(model),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as err:
        return {"text": f"error: could not start nanocode child: {err}", "exitCode": 1, "stderr": str(err)}
    except subprocess.TimeoutExpired as err:
        output = ((err.stdout or "") + "\n" + (err.stderr or "")).strip()
        return {"text": truncate_middle(output or "nanocode child timed out", 18_000, "child output truncated"), "exitCode": 124, "stderr": err.stderr or ""}
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    child_text, data = parse_nanocode_json_output(proc.stdout, proc.stderr)
    if proc.returncode != 0 and proc.stderr:
        child_text = (child_text + "\n" + proc.stderr.strip()).strip()
    return {
        "text": truncate_middle(child_text, 18_000, "child output truncated"),
        "exitCode": proc.returncode,
        "stderr": proc.stderr or "",
        "messages": (data or {}).get("messages", []),
    }


def start_nanocode_job(prompt, tools=None, append_system_prompt="", cwd=None, model=None, auto_emit=True):
    global NANOCODE_JOB_COUNTER
    command, temp_path = nanocode_command(prompt, tools, append_system_prompt, model)
    with NANOCODE_JOB_LOCK:
        NANOCODE_JOB_COUNTER += 1
        job_id = f"nano-{NANOCODE_JOB_COUNTER}"

    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd or os.getcwd(),
            env=nanocode_child_env(model),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise

    job = {
        "id": job_id,
        "proc": proc,
        "prompt": prompt,
        "cwd": cwd or os.getcwd(),
        "model": model or MODEL,
        "tools": tools if tools is not None else "unified",
        "startedAt": time.time(),
        "stdout": [],
        "stderr": [],
        "tempPath": temp_path,
        "cleaned": False,
        "autoEmit": auto_emit,
        "emitted": False,
        "completedAt": None,
    }

    def drain(stream, target):
        try:
            for line in stream:
                target.append(line)
        except Exception as err:
            target.append(f"[reader error: {err}]\n")

    stdout_thread = threading.Thread(target=drain, args=(proc.stdout, job["stdout"]), daemon=True)
    stderr_thread = threading.Thread(target=drain, args=(proc.stderr, job["stderr"]), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    job["stdoutThread"] = stdout_thread
    job["stderrThread"] = stderr_thread

    def monitor():
        returncode = proc.wait()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        job["completedAt"] = time.time()
        cleanup_nanocode_job(job)
        if job.get("autoEmit") and not job.get("emitted"):
            job["emitted"] = True
            NANOCODE_JOB_EVENTS.put(build_nanocode_job_event(job, returncode))

    with NANOCODE_JOB_LOCK:
        NANOCODE_JOBS[job_id] = job
    threading.Thread(target=monitor, daemon=True).start()
    return job


def cleanup_nanocode_job(job):
    if job.get("cleaned") or job["proc"].poll() is None:
        return
    temp_path = job.get("tempPath")
    if temp_path:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    job["cleaned"] = True


def nanocode_job_text(job, max_chars=18_000):
    stdout = "".join(job.get("stdout", []))
    stderr = "".join(job.get("stderr", []))
    text, _data = parse_nanocode_json_output(stdout, stderr)
    return truncate_middle(text, max_chars, "job output truncated")


def build_nanocode_job_event(job, returncode=None):
    if returncode is None:
        returncode = job["proc"].poll()
    status = "done" if returncode == 0 else "failed"
    elapsed = int((job.get("completedAt") or time.time()) - job["startedAt"])
    stderr = "".join(job.get("stderr", [])).strip()
    output = nanocode_job_text(job, NANOCODE_JOB_AUTO_EMIT_MAX_CHARS)
    if stderr and returncode not in (None, 0):
        output = (output + "\n\nstderr:\n" + truncate_middle(stderr, 4_000, "stderr truncated")).strip()
    return {
        "jobId": job["id"],
        "status": status,
        "exitCode": returncode,
        "elapsedSeconds": elapsed,
        "cwd": job["cwd"],
        "model": job["model"],
        "tools": job["tools"],
        "prompt": job["prompt"],
        "output": output,
    }


def format_nanocode_job_event(event):
    title = f"nanocode job {event['jobId']} {event['status']}"
    lines = [
        f"{title} (exit {event['exitCode']}, {event['elapsedSeconds']}s)",
        f"prompt: {truncate_middle(event['prompt'], 300, 'prompt truncated')}",
        "",
        event.get("output") or "(no output)",
    ]
    return "\n".join(lines)


def nanocode_job_event_message(event):
    return {
        "role": "user",
        "content": (
            f"<nanocode_job_result id={json.dumps(event['jobId'])} status={json.dumps(event['status'])} "
            f"exitCode={json.dumps(event['exitCode'])} elapsedSeconds={json.dumps(event['elapsedSeconds'])}>\n"
            f"<prompt>\n{event['prompt']}\n</prompt>\n"
            f"<output>\n{event.get('output') or '(no output)'}\n</output>\n"
            "</nanocode_job_result>"
        ),
    }


def drain_nanocode_job_events(messages=None, print_events=True):
    drained = []
    while True:
        try:
            event = NANOCODE_JOB_EVENTS.get_nowait()
        except queue.Empty:
            break
        drained.append(event)
        if messages is not None:
            messages.append(nanocode_job_event_message(event))
        if print_events:
            print(f"\n{YELLOW}◌ Background job complete{RESET}")
            print(f"{DIM}{format_nanocode_job_event(event)}{RESET}")
    return drained


def nanocode_job_status(job, include_output=False, max_chars=18_000):
    proc = job["proc"]
    returncode = proc.poll()
    cleanup_nanocode_job(job)
    status = "running" if returncode is None else ("done" if returncode == 0 else "failed")
    elapsed = int(time.time() - job["startedAt"])
    stdout = "".join(job.get("stdout", []))
    event_count = sum(1 for line in stdout.splitlines() if line.strip().startswith("{"))
    lines = [
        f"job {job['id']}: {status}",
        f"pid: {proc.pid}",
        f"exitCode: {returncode if returncode is not None else 'running'}",
        f"elapsedSeconds: {elapsed}",
        f"events: {event_count}",
        f"cwd: {job['cwd']}",
        f"tools: {job['tools']}",
        f"prompt: {truncate_middle(job['prompt'], 500, 'prompt truncated')}",
    ]
    stderr = "".join(job.get("stderr", [])).strip()
    if stderr and returncode not in (None, 0):
        lines.extend(["", "stderr:", truncate_middle(stderr, max_chars, "stderr truncated")])
    if include_output or returncode is not None:
        output = nanocode_job_text(job, max_chars)
        heading = "partial output:" if returncode is None else "output:"
        lines.extend(["", heading, output])
    return "\n".join(lines)


def list_nanocode_jobs():
    if not NANOCODE_JOBS:
        return "No nanocode jobs."
    lines = []
    for job_id, job in sorted(NANOCODE_JOBS.items()):
        proc = job["proc"]
        returncode = proc.poll()
        cleanup_nanocode_job(job)
        status = "running" if returncode is None else ("done" if returncode == 0 else "failed")
        elapsed = int(time.time() - job["startedAt"])
        lines.append(f"{job_id}: {status}, pid={proc.pid}, exit={returncode if returncode is not None else 'running'}, elapsed={elapsed}s, prompt={job['prompt'][:80]}")
    return "\n".join(lines)


def cancel_nanocode_job(job):
    proc = job["proc"]
    if proc.poll() is not None:
        cleanup_nanocode_job(job)
        return f"job {job['id']} already finished with exitCode {proc.returncode}"
    job["autoEmit"] = False
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    cleanup_nanocode_job(job)
    return f"job {job['id']} cancelled with exitCode {proc.returncode}"


def nanocode_child_tool(args):
    prompt = args.get("prompt") or args.get("task") or ""
    background = args.get("background")
    if isinstance(background, str):
        background = background.lower() not in ("0", "false", "no")
    auto_emit = args.get("autoEmit", True)
    if isinstance(auto_emit, str):
        auto_emit = auto_emit.lower() not in ("0", "false", "no")
    action = args.get("action")
    if not action:
        action = "start" if prompt.strip() and background is True else ("run" if prompt.strip() else "list")
    action = action.lower()
    max_chars = clamp_number(args.get("maxCharacters"), 18_000, 1_000, 100_000)

    if action == "list":
        return list_nanocode_jobs()

    job_id = args.get("jobId") or args.get("id")
    if action in ("status", "read", "cancel"):
        if not job_id:
            return "error: jobId is required"
        job = NANOCODE_JOBS.get(job_id)
        if not job:
            return f"error: unknown nanocode job {job_id}"
        if action == "cancel":
            return cancel_nanocode_job(job)
        return nanocode_job_status(job, include_output=(action == "read"), max_chars=max_chars)

    if action not in ("run", "start"):
        return "error: action must be run, start, list, status, read, or cancel"

    if not prompt.strip():
        return "error: prompt is required"

    if action == "run":
        result = run_nanocode_process(
            prompt,
            tools=args.get("tools"),
            append_system_prompt=args.get("appendSystemPrompt", ""),
            cwd=args.get("cwd"),
            model=args.get("model"),
            timeout_seconds=args.get("timeoutSeconds", 120),
        )
        prefix = "" if result["exitCode"] == 0 else f"nanocode child exited {result['exitCode']}\n"
        return prefix + result["text"]

    try:
        job = start_nanocode_job(
            prompt,
            tools=args.get("tools"),
            append_system_prompt=args.get("appendSystemPrompt", ""),
            cwd=args.get("cwd"),
            model=args.get("model"),
            auto_emit=bool(auto_emit),
        )
    except Exception as err:
        return f"error: could not start nanocode job: {err}"

    return (
        f"Started nanocode job {job['id']} (pid {job['proc'].pid}).\n"
        f"It will auto-emit its result into the parent session when it finishes. "
        f"Do not poll this job repeatedly in the same response unless the user asked you to wait. "
        f"Tell the user the job id. Later use action=status jobId={job['id']}, "
        f"action=read, or action=cancel if needed."
    )


# =============================================================================
# 7. Unified tool registry and prompt exposure
# =============================================================================
# There is one default tool surface. --tools is only an advanced override for
# smoke tests/headless constraints, not a split core/extended product mode.


def schema_param(param):
    if isinstance(param, dict):
        return param
    return {"type": "number" if param == "number" else param}


def make_function_schema(name, spec):
    params = spec["params"]
    if isinstance(params, dict) and params.get("type") == "object":
        parameters = params
    else:
        properties, required = {}, []
        for param_name, param_type in params.items():
            optional = isinstance(param_type, dict) or (
                isinstance(param_type, str) and param_type.endswith("?")
            )
            base_type = param_type.rstrip("?") if isinstance(param_type, str) else param_type
            schema = schema_param(base_type)
            if isinstance(schema, dict) and "required" in schema:
                property_required = bool(schema.pop("required"))
                optional = not property_required
            properties[param_name] = schema
            if not optional:
                required.append(param_name)
        parameters = {"type": "object", "properties": properties, "required": required}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": spec["description"],
            "parameters": parameters,
        },
    }


def base_tool_specs():
    return {
        "read": {
            "description": "Read the contents of a file. Supports text files. Output is truncated to 2000 lines or 50KB. Use offset/limit for large files. Offset is 1-indexed.",
            "params": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
            "fn": read,
            "snippet": "read: read file contents with line numbers.",
            "guidelines": ["Use read to examine files instead of cat or sed."],
        },
        "write": {
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
            "params": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
            "fn": write,
            "snippet": "write: create or overwrite a file.",
            "guidelines": ["Use write only for new files or complete rewrites."],
        },
        "edit": {
            "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits.",
            "params": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {"type": "array", "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.", "items": {"type": "object", "properties": {"oldText": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call."}, "newText": {"type": "string", "description": "Replacement text for this targeted edit."}}, "required": ["oldText", "newText"], "additionalProperties": False}},
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
            "fn": edit,
            "snippet": "edit: precise string replacement in files.",
            "guidelines": [
                "Use edit for precise changes (edits[].oldText must match exactly).",
                "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls.",
                "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
                "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
            ],
        },
        "glob": {
            "description": "Find files by pattern, sorted by mtime",
            "params": {"pat": "string", "path": "string?"},
            "fn": glob,
            "snippet": "glob: find files by glob pattern.",
        },
        "find": {
            "description": "Find files by pattern under a path",
            "params": {"pattern": "string?", "pat": "string?", "path": "string?", "limit": "number?", "hidden": "boolean?"},
            "fn": find_files,
            "snippet": "find: find files by pattern under a path.",
        },
        "ls": {
            "description": "List files and directories in a path",
            "params": {"path": "string?", "limit": "number?"},
            "fn": ls,
            "snippet": "ls: list files and directories.",
        },
        "grep": {
            "description": "Search files for regex pattern",
            "params": {"pat": "string", "path": "string?"},
            "fn": grep,
            "snippet": "grep: search files by regex.",
        },
        "bash": {
            "description": "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to the last 2000 lines or 50KB. Optionally provide a timeout in seconds.",
            "params": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
                },
                "required": ["command"],
            },
            "fn": bash,
            "snippet": "bash: run shell commands.",
            "guidelines": [
                "Use bash for file operations like ls, grep, find, and for project commands/tests.",
                "Bash commands run with mise hot-reload: nanocode refreshes the environment with `mise env -s bash` before each bash execution.",
            ],
        },
        "nanocode": {
            "description": "Run an isolated headless nanocode process, or manage explicit background jobs.",
            "params": {
                "action": {"type": "string", "enum": ["run", "start", "list", "status", "read", "cancel"], "description": "Action. Defaults to run when prompt is present, otherwise list. Use start for background jobs."},
                "prompt": {"type": "string", "description": "Task for a headless nanocode process"},
                "jobId": {"type": "string", "description": "Job id for status/read/cancel"},
                "background": {"type": "boolean", "description": "When action is omitted, true starts a background job; otherwise prompt defaults to synchronous run."},
                "autoEmit": {"type": "boolean", "description": "For action=start, automatically inject the completed job result into the parent session. Default: true."},
                "tools": {"type": "string", "description": "Advanced override: none, all/default, or comma-separated tool names. Omitted uses the same unified tool set as the parent."},
                "appendSystemPrompt": {"type": "string", "description": "Extra system prompt for the child process"},
                "cwd": {"type": "string", "description": "Working directory for the child process"},
                "model": {"type": "string", "description": "Optional model override for the child process"},
                "timeoutSeconds": {"type": "integer", "description": "Synchronous child timeout, max 600 seconds"},
                "maxCharacters": {"type": "integer", "description": "Maximum job output returned for read/status"},
            },
            "fn": nanocode_child_tool,
            "snippet": "nanocode: run a headless child nanocode process; use action=start only when an explicit background job is needed.",
            "guidelines": [
                "Use nanocode for isolated exploration or side work that would add noise to the parent context.",
                "Default to a synchronous run so the user gets one clean result. Use action=start for background only when the user asks to keep working while it runs.",
                "After starting a background job, do not poll repeatedly in the same turn unless the user explicitly asks you to wait; report the job id and stop.",
                "Child processes use the same unified tool set by default; pass tools=none or an explicit comma-separated allowlist only when the user asks for a constrained run.",
            ],
        },
        "webfetch": {
            "description": "Fetch an HTTP(S) URL and return readable text.",
            "params": {
                "url": {"type": "string", "description": "HTTP or HTTPS URL to fetch", "required": True},
                "format": {"type": "string", "enum": ["markdown", "text", "html"], "description": "Output format. Defaults to markdown for HTML, text for non-HTML text."},
                "timeoutSeconds": {"type": "integer", "description": "Request timeout in seconds, max 120"},
                "maxCharacters": {"type": "integer", "description": "Maximum characters returned to the model"},
            },
            "fn": webfetch,
            "snippet": "webfetch: fetch a URL and return text, markdown, or raw HTML for inspection.",
            "guidelines": [
                "Use webfetch when the user gives a URL or a search result needs direct inspection.",
                "Prefer format=markdown for HTML pages unless raw HTML is required.",
            ],
        },
        "websearch": {
            "description": "Search the web for current information. Uses Exa MCP when available and falls back to DuckDuckGo HTML search.",
            "params": {
                "query": {"type": "string", "description": "Search query", "required": True},
                "numResults": {"type": "integer", "description": "Number of results, default 8, max 10"},
                "provider": {"type": "string", "enum": ["auto", "exa", "duckduckgo"], "description": "Search provider. auto tries Exa then DuckDuckGo."},
                "type": {"type": "string", "enum": ["auto", "fast", "deep"], "description": "Exa search mode"},
                "livecrawl": {"type": "string", "enum": ["fallback", "preferred", "always", "never"], "description": "Exa livecrawl mode"},
                "contextMaxCharacters": {"type": "integer", "description": "Maximum Exa context characters"},
            },
            "fn": websearch,
            "snippet": "websearch: search the live web for current information, returning titles, URLs, and snippets/context.",
            "guidelines": [
                "Use websearch when the answer may depend on recent, external, or URL-discoverable information.",
                "Use webfetch after websearch when a source page needs direct inspection.",
            ],
        },
        "mcp_search": {
            "description": "Search configured MCP servers for available tools without calling them.",
            "params": {"query": "string?", "limit": "number?", "refresh": "boolean?"},
            "fn": mcp_search,
            "snippet": "mcp_search: find tools exposed by MCP servers from ~/.nanocode/mcp.json layered with project .nanocode/mcp.json. Use refresh=true after config changes.",
            "guidelines": [
                "Prefer MCP over websearch when the request matches a configured MCP server domain.",
                "For Google Cloud, Cloud Run, Firebase, Android, Chrome, Go, Gemini, TensorFlow, or web.dev documentation, prefer google-developer-knowledge MCP before websearch.",
                "Use direct mcp__server__tool surfaces when they are available; otherwise use mcp_search, then mcp_inspect, then mcp_call.",
                "Do not guess MCP tool arguments; inspect the schema first when a direct surface schema is not enough.",
            ],
        },
        "mcp_inspect": {
            "description": "Inspect one MCP tool schema and description before calling it.",
            "params": {"server": "string?", "tool": "string", "refresh": "boolean?"},
            "fn": mcp_inspect,
            "snippet": "mcp_inspect: inspect the schema for one MCP tool.",
        },
        "mcp_call": {
            "description": "Call a configured MCP tool by server and tool name.",
            "params": {
                "server": {"type": "string", "description": "Configured MCP server name", "required": True},
                "tool": {"type": "string", "description": "MCP tool name to call", "required": True},
                "arguments": {"type": "object", "description": "Arguments object matching the MCP tool input schema"},
            },
            "fn": mcp_call,
            "snippet": "mcp_call: execute a tool from a configured MCP server after inspecting its schema.",
        },
    }


def tool_specs():
    specs = base_tool_specs()
    specs.update(direct_mcp_specs(os.getcwd()))
    return specs


def make_tools(tool_names=None):
    return [make_function_schema(name, spec) for name, spec in selected_tool_specs(tool_names).items()]


def tool_prompt(specs):
    lines = ["Available tools:"]
    for name, spec in specs.items():
        lines.append(f"- {name}: {spec['description'].splitlines()[0]}")
    guidelines = []
    for spec in specs.values():
        guidelines.extend(spec.get("guidelines", []))
    if guidelines:
        lines.extend(["", "Tool guidance:"])
        for item in dict.fromkeys(guidelines):
            lines.append(f"- {item}")
    return "\n".join(lines)


def run_tool(name, args):
    try:
        specs = tool_specs()
        if name not in specs:
            return f"error: unknown tool {name}"
        return text_from_tool_result(specs[name]["fn"](args or {}))
    except Exception as err:
        return f"error: {err}"


# =============================================================================
# 8. OpenAI-compatible streaming chat and reasoning extraction
# =============================================================================


def event_data_lines(response):
    for raw_line in response:
        line = raw_line.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data:
            yield data


def merge_tool_call(tool_calls, delta_call):
    idx = delta_call.get("index", len(tool_calls))
    while len(tool_calls) <= idx:
        tool_calls.append(
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )

    tool_call = tool_calls[idx]
    if delta_call.get("id"):
        tool_call["id"] = delta_call["id"]
    if delta_call.get("type"):
        tool_call["type"] = delta_call["type"]

    delta_function = delta_call.get("function") or {}
    function = tool_call.setdefault("function", {"name": "", "arguments": ""})
    if delta_function.get("name"):
        function["name"] += delta_function["name"]
    if delta_function.get("arguments"):
        function["arguments"] += delta_function["arguments"]


def is_reasoning_part(part):
    part_type = part.get("type", "") if isinstance(part, dict) else ""
    return "reason" in part_type or part_type == "thinking"


def part_text(part):
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    text = part.get("text", part.get("content", ""))
    return text if isinstance(text, str) else ""


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part_text(part) for part in content if not is_reasoning_part(part)
        )
    return ""


def reasoning_text(message):
    parts = []
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)

    for detail in message.get("reasoning_details", []) or []:
        text = part_text(detail)
        if text.strip():
            parts.append(text)

    content = message.get("content")
    if isinstance(content, list):
        parts.extend(part_text(part) for part in content if is_reasoning_part(part))

    unique = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return "\n".join(unique).strip()


def delta_reasoning(delta):
    for key in ("reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return key, value

    details = []
    for detail in delta.get("reasoning_details", []) or []:
        text = part_text(detail)
        if text:
            details.append(text)
    if details:
        return "reasoning", "".join(details)
    return None, ""


def stream_response(response, display=True, event_callback=None):
    role = "assistant"
    content_parts = []
    reasoning_parts = {"reasoning": [], "reasoning_content": []}
    tool_calls = []
    mode = None

    def show_thinking_delta(text):
        nonlocal mode
        if not display or not SHOW_THINKING or not text:
            return
        if mode != "thinking":
            print(f"\n{YELLOW}◌ Thinking{RESET}\n{DIM}", end="", flush=True)
            mode = "thinking"
        print(text, end="", flush=True)

    def show_content_delta(text):
        nonlocal mode
        if not display or not text:
            return
        if mode == "thinking":
            print(f"{RESET}\n", end="", flush=True)
        if mode != "answer":
            print(f"\n{CYAN}⏺{RESET} ", end="", flush=True)
            mode = "answer"
        print(text, end="", flush=True)

    for data in event_data_lines(response):
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        if delta.get("role"):
            role = delta["role"]

        reasoning_key, reasoning = delta_reasoning(delta)
        if reasoning_key and reasoning:
            reasoning_parts[reasoning_key].append(reasoning)
            if event_callback:
                event_callback({"type": "reasoning_delta", "delta": reasoning})
            show_thinking_delta(reasoning)

        content = delta.get("content")
        if isinstance(content, list):
            content = content_text(content)
        if isinstance(content, str) and content:
            content_parts.append(content)
            if event_callback:
                event_callback({"type": "text_delta", "delta": content})
            show_content_delta(content)

        for delta_call in delta.get("tool_calls", []) or []:
            merge_tool_call(tool_calls, delta_call)

    if display:
        if mode == "thinking":
            print(RESET)
        elif mode == "answer":
            print()

    for idx, tool_call in enumerate(tool_calls):
        if not tool_call.get("id"):
            tool_call["id"] = f"call_{idx}"

    message = {"role": role, "content": "".join(content_parts)}
    for key, parts in reasoning_parts.items():
        if parts:
            message[key] = "".join(parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"_streamed": True, "choices": [{"message": message}]}


def api_headers(accept="text/event-stream"):
    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
        "User-Agent": os.environ.get("NANOCODE_USER_AGENT", "nanocode/1.0"),
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def handle_http_error(err):
    detail = err.read().decode("utf-8", "replace").strip()
    hint = ""
    if err.code in (401, 403) and "opencode.ai/zen/go" in API_URL:
        if not API_KEY:
            hint = " (no OpenCode Go API key found)"
        else:
            hint = f" (auth source: {API_KEY_SOURCE})"
    raise RuntimeError(f"HTTP {err.code} from {API_URL}: {detail}{hint}") from None


def call_api(messages, system_prompt, tool_names=None, display=True, event_callback=None):
    payload = {
        "model": MODEL,
        "max_tokens": 8192,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "stream": True,
    }
    tools = make_tools(tool_names)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers=api_headers("text/event-stream"),
    )
    try:
        response = urllib.request.urlopen(request)
        return stream_response(response, display, event_callback)
    except urllib.error.HTTPError as err:
        handle_http_error(err)


def complete_once(question, system_prompt):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 2048,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
            }
        ).encode(),
        headers=api_headers("application/json"),
    )
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as err:
        handle_http_error(err)
    message = data.get("choices", [{}])[0].get("message", {})
    return content_text(message.get("content")) or reasoning_text(message) or "(no output)"


# =============================================================================
# 9. Context compaction
# =============================================================================
# Uses a cheap local token estimate to decide when to summarize older turns.
# The trigger mirrors Pi's shape: compact before the remaining context falls
# below a reserve token budget / remaining-context percentage.


def estimate_tokens_text(text):
    # Cheap, deterministic approximation. Good enough for deciding when to compact.
    return max(1, (len(text or "") + 3) // 4)


def message_text_for_tokens(message):
    parts = [message.get("role", "")]
    parts.append(content_text(message.get("content")) if not isinstance(message.get("content"), str) else message.get("content", ""))
    if message.get("reasoning"):
        parts.append(str(message.get("reasoning")))
    if message.get("reasoning_content"):
        parts.append(str(message.get("reasoning_content")))
    for tool_call in message.get("tool_calls", []) or []:
        parts.append(json.dumps(tool_call, ensure_ascii=False))
    return "\n".join(part for part in parts if part)


def estimate_messages_tokens(messages):
    return sum(estimate_tokens_text(message_text_for_tokens(message)) for message in messages)


def compaction_remaining_budget():
    percent_budget = int(CONTEXT_WINDOW_TOKENS * COMPACTION_REMAINING_RATIO)
    return max(COMPACTION_RESERVE_TOKENS, percent_budget)


def context_budget_status(messages, system_prompt=""):
    system_tokens = estimate_tokens_text(system_prompt) if system_prompt else 0
    message_tokens = estimate_messages_tokens(messages)
    used = system_tokens + message_tokens
    remaining = max(0, CONTEXT_WINDOW_TOKENS - used)
    ratio = used / max(1, CONTEXT_WINDOW_TOKENS)
    budget = compaction_remaining_budget()
    return {
        "used": used,
        "system": system_tokens,
        "messages": message_tokens,
        "remaining": remaining,
        "ratio": ratio,
        "budget": budget,
        "triggerAt": max(0, CONTEXT_WINDOW_TOKENS - budget),
    }


def should_auto_compact(messages, system_prompt=""):
    if not COMPACTION_ENABLED:
        return False
    status = context_budget_status(messages, system_prompt)
    return status["used"] > status["triggerAt"] or status["remaining"] <= status["budget"]


def serialize_message_for_compaction(message):
    role = message.get("role", "unknown")
    content = content_text(message.get("content")) if not isinstance(message.get("content"), str) else message.get("content", "")
    lines = [f"[{role.title()}]: {truncate_middle(content, 2000, 'message truncated')}" if content else f"[{role.title()}]"]
    thinking = reasoning_text(message)
    if thinking:
        lines.append(f"[Assistant thinking]: {truncate_middle(thinking, 1200, 'thinking truncated')}")
    tool_calls = message.get("tool_calls", []) or []
    if tool_calls:
        rendered = []
        for call in tool_calls:
            fn = call.get("function", {})
            rendered.append(f"{fn.get('name', '')}({truncate_middle(fn.get('arguments', ''), 800, 'args truncated')})")
        lines.append("[Assistant tool calls]: " + "; ".join(rendered))
    return "\n".join(lines)


def serialize_conversation(messages):
    return "\n\n".join(serialize_message_for_compaction(message) for message in messages)


def previous_compaction_summary(messages):
    if not messages:
        return ""
    first = messages[0]
    content = first.get("content") if first.get("role") == "user" else ""
    if isinstance(content, str) and content.startswith("<compaction_summary"):
        return content
    return ""


def compaction_cut_index(messages, keep_recent_tokens=COMPACTION_KEEP_RECENT_TOKENS):
    if len(messages) < 4:
        return 0
    tokens = 0
    cut = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        tokens += estimate_tokens_text(message_text_for_tokens(messages[idx]))
        cut = idx
        if tokens >= keep_recent_tokens:
            break
    while cut > 0 and messages[cut].get("role") != "user":
        cut -= 1
    if cut <= 0:
        return 0
    return cut


def compact_messages(messages, system_prompt, instructions=""):
    cut = compaction_cut_index(messages)
    if cut <= 0:
        return messages, False, "not enough complete conversation history to compact"

    old_messages = messages[:cut]
    kept_messages = messages[cut:]
    previous_summary = previous_compaction_summary(old_messages)
    conversation = serialize_conversation(old_messages)
    focus = f"\nAdditional compaction instructions:\n{instructions}\n" if instructions.strip() else ""
    prompt = f"""Summarize this nanocode conversation for future continuation.

Use this exact structure:

## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context
<read-files>
</read-files>
<modified-files>
</modified-files>

Preserve file paths, commands run, decisions, current task state, and anything needed to continue safely. Be concise but complete.{focus}

Previous summary, if any:
{previous_summary or '(none)'}

Conversation to summarize:
{conversation}
"""
    summary = complete_once(prompt, "You are nanocode's compaction summarizer. Return only the structured summary.")
    tokens_before = context_budget_status(messages, system_prompt)["used"]
    summary_message = {
        "role": "user",
        "content": f"<compaction_summary tokensBefore={tokens_before} firstKeptIndex={cut}>\n{summary}\n</compaction_summary>",
    }
    return [summary_message, *kept_messages], True, f"compacted {cut} old messages; kept {len(kept_messages)} recent messages"


# =============================================================================
# 10. Interactive rendering, context files, slash commands, and sidecars
# =============================================================================


def separator():
    return f"{DIM}{'─' * min(shutil.get_terminal_size((80, 20)).columns, 80)}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def print_thinking(text):
    if not text:
        return
    print(f"\n{YELLOW}◌ Thinking{RESET}")
    print(f"{DIM}{render_markdown(text)}{RESET}")


CONTEXT_FILE_NAMES = ("AGENTS.md", "agents.md", "CLAUDE.md", "claude.md")
SYSTEM_FILE_NAMES = ("SYSTEM.md", "system.md")
APPEND_SYSTEM_FILE_NAMES = ("APPEND_SYSTEM.md", "append_system.md", "append-system.md")


def selected_tool_specs(tool_names=None):
    specs = tool_specs()
    if tool_names is None:
        return specs
    allowed = set(tool_names)
    return {name: spec for name, spec in specs.items() if name in allowed}


def first_existing(directory, names):
    for name in names:
        path = Path(directory) / name
        if path.is_file():
            return path
    return None


def read_text_file(path):
    try:
        return Path(path).read_text()
    except Exception:
        return ""


def load_context_files(cwd):
    files = []
    seen = set()

    for name in CONTEXT_FILE_NAMES:
        path = AGENT_DIR / name
        if path.is_file():
            real = str(path.resolve())
            if real not in seen:
                files.append({"path": str(path), "content": read_text_file(path)})
                seen.add(real)

    for directory in ancestor_dirs(cwd):
        for name in CONTEXT_FILE_NAMES:
            path = directory / ".nanocode" / name
            if path.is_file():
                real = str(path.resolve())
                if real not in seen:
                    files.append({"path": str(path), "content": read_text_file(path)})
                    seen.add(real)
    return [item for item in files if item["content"].strip()]


def nearest_project_system_file(cwd, names):
    path = Path(cwd).resolve()
    while True:
        candidate = first_existing(path / ".nanocode", names)
        if candidate:
            return candidate
        if path.parent == path:
            return None
        path = path.parent


def load_system_prompt_file(cwd):
    project = nearest_project_system_file(cwd, SYSTEM_FILE_NAMES)
    if project:
        return {"path": str(project), "content": read_text_file(project)}
    global_file = first_existing(AGENT_DIR, SYSTEM_FILE_NAMES)
    if global_file:
        return {"path": str(global_file), "content": read_text_file(global_file)}
    return None


def load_append_system_files(cwd):
    files = []
    seen = set()
    global_file = first_existing(AGENT_DIR, APPEND_SYSTEM_FILE_NAMES)
    if global_file:
        files.append({"path": str(global_file), "content": read_text_file(global_file)})
        seen.add(str(global_file.resolve()))
    for directory in ancestor_dirs(cwd):
        project_file = first_existing(directory / ".nanocode", APPEND_SYSTEM_FILE_NAMES)
        if project_file:
            real = str(project_file.resolve())
            if real not in seen:
                files.append({"path": str(project_file), "content": read_text_file(project_file)})
                seen.add(real)
    return [item for item in files if item["content"].strip()]


def format_loaded_files(title, files, tag):
    if not files:
        return ""
    parts = [title]
    for item in files:
        parts.append(f"<{tag} path={json.dumps(item['path'])}>\n{item['content'].strip()}\n</{tag}>")
    return "\n\n".join(parts)


def resolve_prompt_arg(value):
    if value is None:
        return ""
    text = str(value)
    path = Path(text).expanduser()
    if path.is_file():
        return read_text_file(path)
    return text


def build_system_prompt(cwd, skills, tool_names=None, system_prompt_override=None, append_system_prompts=None, include_context=True):
    file_override = load_system_prompt_file(cwd) if system_prompt_override is None else None
    override_text = system_prompt_override if system_prompt_override is not None else (file_override or {}).get("content")

    if override_text and override_text.strip():
        prompt = [override_text.strip()]
    else:
        prompt = [
            f"Concise coding assistant. cwd: {cwd}",
            f"Current date: {time.strftime('%Y-%m-%d')}",
            "Visible thinking should be brief progress notes, not a transcript of indecision. Do not narrate repeated waiting/polling.",
            "",
            tool_prompt(selected_tool_specs(tool_names)),
            "",
            "Mise tool hot-reload is enabled for bash commands. Use bash normally; do not hard-code mise install paths.",
        ]

    if include_context:
        context_block = format_loaded_files("Context files loaded from .nanocode/AGENTS.md / .nanocode/CLAUDE.md:", load_context_files(cwd), "context_file")
        if context_block:
            prompt.extend(["", context_block])

    append_files = load_append_system_files(cwd)
    append_blocks = []
    if append_files:
        append_blocks.append(format_loaded_files("Append-system files:", append_files, "append_system_file"))
    for item in append_system_prompts or []:
        text = resolve_prompt_arg(item).strip()
        if text:
            append_blocks.append(text)
    if append_blocks:
        prompt.extend(["", "\n\n".join(append_blocks)])

    skills_prompt = format_skills_for_prompt(skills)
    if skills_prompt:
        prompt.extend(["", skills_prompt])
    return "\n".join(prompt).strip()


def build_btw_prompt(request):
    tool_line = (
        "You may use the provided nanocode tools if they materially help, but keep this sidecar non-mutating."
        if request["useTools"]
        else "Do not use tools. Answer from the model's own context only."
    )
    return f"""You are a throw-away read-only sidecar in the same workspace as the parent nanocode session.

Answer the user's BTW question directly and briefly. {tool_line}

Rules:
- Do not modify files.
- Do not run shell commands.
- Do not attempt session changes.
- Keep the final answer compact and useful.

BTW question:
{request['question']}"""


def parse_btw_request(args):
    raw = args.strip()
    if raw.startswith("--tools "):
        return {"useTools": True, "question": raw[len("--tools ") :].strip()}
    if raw.startswith("-t "):
        return {"useTools": True, "question": raw[len("-t ") :].strip()}
    return {"useTools": False, "question": raw}


def run_sidecar_agent(question, system_prompt, tool_names):
    local_messages = [{"role": "user", "content": question}]
    final_text = ""
    for _ in range(12):
        response = call_api(local_messages, system_prompt, tool_names=tool_names, display=False)
        message = response.get("choices", [{}])[0].get("message", {})
        text = content_text(message.get("content"))
        final_text = text or final_text
        tool_calls = message.get("tool_calls") or []

        assistant_message = {"role": "assistant", "content": text or ""}
        for key in ("reasoning", "reasoning_content"):
            if message.get(key):
                assistant_message[key] = message[key]
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        local_messages.append(assistant_message)

        if not tool_calls:
            return final_text or "(no output)"

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args, parse_error = parse_tool_args(tool_call)
            result = (
                f"error: {parse_error}"
                if parse_error
                else run_tool(tool_name, tool_args)
            )
            local_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": result,
                }
            )
    return final_text or "BTW sidecar stopped after 12 tool turns."


def run_btw(args):
    request = parse_btw_request(args)
    if not request["question"]:
        return "Usage: /btw <question> or /btw --tools <question>"
    if not request["useTools"]:
        return truncate_middle(complete_once(request["question"], build_btw_prompt(request)), 18_000, "btw truncated")

    answer = run_sidecar_agent(
        request["question"],
        build_btw_prompt(request),
        None,
    )
    return truncate_middle(answer, 18_000, "btw truncated")


def print_skills(skills):
    if not skills:
        print(f"{YELLOW}No skills discovered{RESET}")
        return
    for skill in skills:
        hidden = " hidden" if skill.get("disabled") else ""
        print(f"{GREEN}{skill['name']}{RESET}{DIM}{hidden} — {skill['path']}{RESET}")
        print(f"  {skill['description']}")


def models_api_url():
    if API_URL.endswith("/chat/completions"):
        return API_URL[: -len("/chat/completions")] + "/models"
    return BASE_URL + "/models"


def fetch_available_models():
    request = urllib.request.Request(models_api_url(), headers=api_headers("application/json"))
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read())
    raw_models = data.get("data", data.get("models", [])) if isinstance(data, dict) else []
    model_ids = []
    for item in raw_models:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
        else:
            model_id = None
        if model_id and model_id not in model_ids:
            model_ids.append(str(model_id))
    return sorted(model_ids, key=str.lower)


def format_model_matches(model_ids, limit=40):
    shown = model_ids[:limit]
    lines = [f"- {model_id}" for model_id in shown]
    if len(model_ids) > limit:
        lines.append(f"... +{len(model_ids) - limit} more")
    return "\n".join(lines)


def model_command(args):
    global MODEL
    query = args.strip()
    model_ids = []
    fetch_error = ""
    try:
        model_ids = fetch_available_models()
    except Exception as err:
        fetch_error = str(err)

    if not query:
        lines = [f"Current model: {MODEL}"]
        if model_ids:
            lines.append(f"Available models from {models_api_url()}:")
            lines.append(format_model_matches(model_ids))
            lines.append("Use /model <exact-id> to switch.")
        elif fetch_error:
            lines.append(f"Could not fetch models from {models_api_url()}: {fetch_error}")
            lines.append("Use /model <id> to set a model id directly.")
        else:
            lines.append("No models returned. Use /model <id> to set a model id directly.")
        return "\n".join(lines)

    if model_ids:
        exact = next((model_id for model_id in model_ids if model_id == query), None)
        if not exact:
            exact = next((model_id for model_id in model_ids if model_id.lower() == query.lower()), None)
        if exact:
            MODEL = exact
            return f"Model set to {MODEL} (conversation kept)"

        matches = [model_id for model_id in model_ids if query.lower() in model_id.lower()]
        if matches:
            return (
                f"No exact model id matched {query!r}. Matches:\n"
                f"{format_model_matches(matches)}\n"
                "Use /model <exact-id> to switch."
            )

        MODEL = query
        return f"Model set to {MODEL} (not returned by /models; conversation kept)"

    MODEL = query
    suffix = f"; could not verify against /models: {fetch_error}" if fetch_error else ""
    return f"Model set to {MODEL} (conversation kept{suffix})"


def mcp_status():
    config = load_mcp_config(os.getcwd())
    servers = enabled_servers(config)
    lines = [f"MCP config: {AGENT_DIR / 'mcp.json'}"]
    lines.append(f"Enabled servers: {len(servers)}")
    for name, server in servers:
        selected = selected_model_tools(server)
        direct = f", selectedTools={','.join(selected)}" if selected else ""
        desc = f" - {server.get('description', '')}" if server.get("description") else ""
        lines.append(f"- {name} ({server.get('type') or ('stdio' if server.get('command') else 'remote')}{direct}){desc}")
    return "\n".join(lines)


def parse_tool_args(tool_call):
    try:
        arguments = tool_call.get("function", {}).get("arguments") or "{}"
        args = json.loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be a JSON object")
        return args, None
    except Exception as err:
        return {}, f"invalid tool arguments: {err}"


def show_tool_call(tool_name, tool_args):
    arg_preview = ", ".join(
        f"{key}={str(value)[:40]}" for key, value in list(tool_args.items())[:2]
    ) or "{}"
    print(f"\n{GREEN}⏺ {tool_name}{RESET}({DIM}{arg_preview}{RESET})")


def show_tool_result(result):
    result_lines = result.split("\n")
    preview = result_lines[0][:60]
    if len(result_lines) > 1:
        preview += f" ... +{len(result_lines) - 1} lines"
    elif result_lines and len(result_lines[0]) > 60:
        preview += "..."
    print(f"  {DIM}⎿  {preview}{RESET}")


def assistant_message_for_history(message, text, tool_calls):
    assistant_message = {"role": "assistant", "content": text or ""}
    for key in ("reasoning", "reasoning_content"):
        if message.get(key):
            assistant_message[key] = message[key]
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return assistant_message


# =============================================================================
# 11. Native interactive steering / follow-up queue
# =============================================================================
# These queues apply only to the parent interactive process. They are separate
# from child nanocode jobs and are delivered at safe model/tool boundaries.


def native_queue_for(mode):
    return NATIVE_FOLLOWUP_QUEUE if mode in ("queue", "followup", "followUp") else NATIVE_STEERING_QUEUE


def enqueue_native_message(mode, text):
    text = str(text or "").strip()
    if not text:
        return "error: message is required"
    queue_ref = native_queue_for(mode)
    with NATIVE_QUEUE_LOCK:
        queue_ref.append(text)
        steering_count = len(NATIVE_STEERING_QUEUE)
        followup_count = len(NATIVE_FOLLOWUP_QUEUE)
    label = "follow-up" if queue_ref is NATIVE_FOLLOWUP_QUEUE else "steering"
    return f"Queued {label} message ({steering_count} steering, {followup_count} follow-up)."


def pop_native_message(mode):
    queue_ref = native_queue_for(mode)
    with NATIVE_QUEUE_LOCK:
        if not queue_ref:
            return None
        return queue_ref.pop(0)


def get_native_queued_items():
    with NATIVE_QUEUE_LOCK:
        return [
            *({"mode": "steer", "text": text} for text in NATIVE_STEERING_QUEUE),
            *({"mode": "queue", "text": text} for text in NATIVE_FOLLOWUP_QUEUE),
        ]


def replace_native_queued_items(items):
    steering, followup = [], []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each queued item must be an object")
        mode = str(item.get("mode", "")).lower()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if mode in ("steer", "steering"):
            steering.append(text)
        elif mode in ("queue", "followup", "follow-up", "follow_up"):
            followup.append(text)
        else:
            raise ValueError(f"unknown queue mode: {item.get('mode')!r}")
    with NATIVE_QUEUE_LOCK:
        NATIVE_STEERING_QUEUE[:] = steering
        NATIVE_FOLLOWUP_QUEUE[:] = followup
    return len(steering), len(followup)


def clear_native_queues():
    with NATIVE_QUEUE_LOCK:
        items = [
            *({"mode": "steer", "text": text} for text in NATIVE_STEERING_QUEUE),
            *({"mode": "queue", "text": text} for text in NATIVE_FOLLOWUP_QUEUE),
        ]
        NATIVE_STEERING_QUEUE.clear()
        NATIVE_FOLLOWUP_QUEUE.clear()
    return items


def format_native_queue(items=None):
    items = get_native_queued_items() if items is None else items
    if not items:
        return "No queued native messages."
    lines = []
    for idx, item in enumerate(items, 1):
        label = "steer" if item["mode"] == "steer" else "queue"
        lines.append(f"{idx}. [{label}] {truncate_middle(item['text'], 500, 'message truncated')}")
    return "\n".join(lines)


def edit_native_queue_in_editor():
    # Hold the queue lock while editing so the native worker cannot deliver a
    # queued message that the user is currently editing/deleting.
    with NATIVE_QUEUE_LOCK:
        items = get_native_queued_items()
        if not items:
            return "No queued native messages to edit."
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            return "No $VISUAL or $EDITOR set. Use /dequeue to clear and print queued messages, then re-queue edited text."
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nanocode-queue.json") as tmp:
            tmp.write(json.dumps(items, indent=2, ensure_ascii=False))
            tmp.write("\n")
            path = tmp.name
        try:
            command = [*shlex.split(editor), path]
            code = subprocess.call(command)
            if code != 0:
                return f"Editor exited with code {code}; queue unchanged."
            edited = json.loads(Path(path).read_text())
            if not isinstance(edited, list):
                return "Edited queue must be a JSON array; queue unchanged."
            steering_count, followup_count = replace_native_queued_items(edited)
            return f"Updated queue ({steering_count} steering, {followup_count} follow-up)."
        except Exception as err:
            return f"Could not edit queue: {err}; queue unchanged."
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass


def deliver_native_queued_message(messages, mode):
    text = pop_native_message(mode)
    if not text:
        return False
    label = "follow-up" if mode in ("queue", "followup", "followUp") else "steering"
    print(f"\n{YELLOW}◌ Delivering queued {label}{RESET}: {DIM}{truncate_middle(text, 300, 'message truncated')}{RESET}")
    messages.append({"role": "user", "content": text})
    return True


# Headless / one-shot agent loop ---------------------------------------------
# Used by --print, --mode json, /btw sidecars, and child nanocode processes.
def run_agent_once(user_input, system_prompt, tool_names=None, display=False, max_turns=12, event_callback=None):
    messages = [{"role": "user", "content": user_input}]
    final_text = ""
    for turn_index in range(max_turns):
        if should_auto_compact(messages, system_prompt):
            messages, compacted, note = compact_messages(messages, system_prompt)
            if event_callback and compacted:
                event_callback({"type": "compaction", "note": note, "context": context_budget_status(messages, system_prompt)})
        if event_callback:
            event_callback({"type": "turn_start", "turn": turn_index + 1})
        response = call_api(messages, system_prompt, tool_names=tool_names, display=display, event_callback=event_callback)
        message = response.get("choices", [{}])[0].get("message", {})
        text = content_text(message.get("content"))
        final_text = text or final_text
        thinking = reasoning_text(message)
        tool_calls = message.get("tool_calls") or []

        if display and not response.get("_streamed"):
            if SHOW_THINKING:
                print_thinking(thinking)
            if text:
                print(f"\n{CYAN}⏺{RESET} {render_markdown(text)}")

        history_message = assistant_message_for_history(message, text, tool_calls)
        messages.append(history_message)
        if event_callback:
            event_callback({"type": "message_end", "role": "assistant", "text": text or "", "toolCalls": tool_calls})
        if not tool_calls:
            return {"text": final_text or "(no output)", "messages": messages}

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args, parse_error = parse_tool_args(tool_call)
            if display:
                show_tool_call(tool_name, tool_args)
            if event_callback:
                event_callback({"type": "tool_start", "name": tool_name, "args": tool_args})
            result = f"error: {parse_error}" if parse_error else run_tool(tool_name, tool_args)
            if display:
                show_tool_result(result)
            if event_callback:
                event_callback({"type": "tool_end", "name": tool_name, "result": result, "isError": bool(parse_error) or result.startswith("error:")})
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": result})
    return {"text": final_text or "agent stopped after max tool turns", "messages": messages}


# Parent interactive turn runner ---------------------------------------------
# Runs in a worker thread so the native REPL can accept steering/follow-up input.
def run_interactive_agent_turn(messages, user_input, system_prompt, tool_names=None, max_turns=100):
    messages.append({"role": "user", "content": user_input})

    for _turn_index in range(max_turns):
        drain_nanocode_job_events(messages)
        if should_auto_compact(messages, system_prompt):
            compacted_messages, compacted, note = compact_messages(messages, system_prompt)
            if compacted:
                messages[:] = compacted_messages
                status = context_budget_status(messages, system_prompt)
                print(f"{YELLOW}◌ Auto-compact: {note}; now ~{status['used']} tokens{RESET}")

        response = call_api(messages, system_prompt, tool_names=tool_names)
        message = response.get("choices", [{}])[0].get("message", {})
        text = content_text(message.get("content"))
        thinking = reasoning_text(message)
        tool_calls = message.get("tool_calls") or []

        if not response.get("_streamed"):
            if SHOW_THINKING:
                print_thinking(thinking)
            if text:
                print(f"\n{CYAN}⏺{RESET} {render_markdown(text)}")

        messages.append(assistant_message_for_history(message, text, tool_calls))

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args, parse_error = parse_tool_args(tool_call)
            show_tool_call(tool_name, tool_args)
            result = f"error: {parse_error}" if parse_error else run_tool(tool_name, tool_args)
            show_tool_result(result)
            messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "content": result})

        drain_nanocode_job_events(messages)
        # Steering wins at every safe boundary, before the next LLM call.
        if deliver_native_queued_message(messages, "steer"):
            continue
        # Follow-ups wait until the agent has no more tool calls.
        if tool_calls:
            continue
        if deliver_native_queued_message(messages, "queue"):
            continue
        return

    print(f"{YELLOW}⏺ Agent stopped after {max_turns} turns; queued messages remain available with /queue{RESET}")


# Main native REPL ------------------------------------------------------------
def main(tool_names=None):
    global SKILLS, MCP_INVENTORY_CACHE, MODEL

    SKILLS = load_skills(os.getcwd())
    system_prompt = build_system_prompt(os.getcwd(), SKILLS, tool_names=tool_names)
    auth = "no auth" if not API_KEY else f"auth: {API_KEY_SOURCE}"
    context_count = len(load_context_files(os.getcwd()))
    print(
        f"{BOLD}nanocode{RESET} | {DIM}{MODEL} ({PROVIDER}, {auth}) | "
        f"{os.getcwd()}{RESET} | {len(SKILLS)} skills | {context_count} context\n"
    )
    messages = []
    agent_state = {"busy": False, "thread": None}
    agent_state_lock = threading.Lock()

    def agent_busy():
        with agent_state_lock:
            thread = agent_state.get("thread")
            if thread is not None and not thread.is_alive():
                agent_state["busy"] = False
            return bool(agent_state.get("busy"))

    def start_native_agent(text):
        if agent_busy():
            print(enqueue_native_message("steer", text))
            return

        def worker():
            try:
                run_interactive_agent_turn(messages, text, system_prompt, tool_names=tool_names)
                drain_nanocode_job_events(messages)
                print()
            except Exception as err:
                print(f"{RED}⏺ Error: {err}{RESET}")
            finally:
                with agent_state_lock:
                    agent_state["busy"] = False

        with agent_state_lock:
            agent_state["busy"] = True
            agent_state["thread"] = threading.Thread(target=worker, daemon=True)
            agent_state["thread"].start()

    while True:
        try:
            # Keep the prompt responsive while the worker thread streams model
            # output. When busy, plain text becomes steering; explicit /queue
            # becomes a follow-up.
            busy = agent_busy()
            if not busy:
                drain_nanocode_job_events(messages)
            print(separator())
            prompt = (
                f"{BOLD}{BLUE}❯{RESET} "
                if not busy
                else f"{BOLD}{YELLOW}❯ working{RESET} "
            )
            user_input = input(prompt).strip()
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break

            # Queue commands are always handled by the native parent process.
            if user_input == "/queue":
                print(format_native_queue())
                continue
            if user_input.startswith("/queue ") or user_input.startswith("/followup "):
                command, _, queued_text = user_input.partition(" ")
                if agent_busy():
                    print(enqueue_native_message("queue", queued_text))
                else:
                    print(f"{DIM}No active agent; running follow-up now.{RESET}")
                    drain_nanocode_job_events(messages)
                    start_native_agent(queued_text)
                continue
            if user_input.startswith("/steer "):
                steer_text = user_input[len("/steer ") :].strip()
                if agent_busy():
                    print(enqueue_native_message("steer", steer_text))
                else:
                    print(f"{DIM}No active agent; running steering message now.{RESET}")
                    drain_nanocode_job_events(messages)
                    start_native_agent(steer_text)
                continue
            if user_input == "/dequeue" or user_input.startswith("/dequeue "):
                arg = user_input[len("/dequeue") :].strip()
                if arg == "edit":
                    print(edit_native_queue_in_editor())
                else:
                    items = clear_native_queues()
                    print("Dequeued native messages:\n" + format_native_queue(items) if items else "No queued native messages.")
                continue

            # While the agent is working, allow safe inspection commands and
            # route all other user text into the steering queue.
            busy = agent_busy()
            if busy:
                if user_input == "/jobs":
                    print(list_nanocode_jobs())
                elif user_input == "/tools":
                    for name in selected_tool_specs(tool_names):
                        print(f"{GREEN}{name}{RESET}")
                elif user_input == "/context":
                    status = context_budget_status(messages, system_prompt)
                    print(
                        f"Context estimate: {status['used']} used / {CONTEXT_WINDOW_TOKENS} "
                        f"window ({status['ratio']:.0%}); queued: {format_native_queue()}"
                    )
                elif user_input == "/model" or user_input.startswith("/model "):
                    print(model_command(user_input[len("/model") :]))
                elif user_input.startswith("/"):
                    print(
                        f"{YELLOW}Agent is working. Plain input queues steering; "
                        f"use /queue <msg> for follow-up, /dequeue to clear, "
                        f"/dequeue edit to edit.{RESET}"
                    )
                else:
                    print(enqueue_native_message("steer", user_input))
                continue

            # Idle-only slash commands can safely mutate session state.
            if user_input == "/c":
                messages.clear()
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue
            if user_input.startswith("/compact"):
                instructions = user_input[len("/compact") :].strip()
                compacted_messages, compacted, note = compact_messages(messages, system_prompt, instructions)
                if compacted:
                    messages[:] = compacted_messages
                color = GREEN if compacted else YELLOW
                print(f"{color}⏺ {note}{RESET}")
                continue
            if user_input == "/context":
                status = context_budget_status(messages, system_prompt)
                print(
                    f"Context estimate: {status['used']} used / {CONTEXT_WINDOW_TOKENS} window "
                    f"({status['ratio']:.0%}), {status['remaining']} remaining "
                    f"({status['system']} system + {status['messages']} messages). "
                    f"Auto-compact when remaining <= {status['budget']} "
                    f"(max of {COMPACTION_RESERVE_TOKENS} tokens and {COMPACTION_REMAINING_RATIO:.0%})."
                )
                continue
            if user_input == "/model" or user_input.startswith("/model "):
                print(model_command(user_input[len("/model") :]))
                continue
            if user_input == "/reload":
                close_mcp_clients()
                MCP_INVENTORY_CACHE = {"key": None, "inventory": None}
                SKILLS = load_skills(os.getcwd())
                system_prompt = build_system_prompt(os.getcwd(), SKILLS, tool_names=tool_names)
                print(
                    f"{GREEN}⏺ Reloaded {len(SKILLS)} skills, "
                    f"{len(load_context_files(os.getcwd()))} context files, jobs, "
                    f"and MCP config{RESET}"
                )
                continue
            if user_input == "/skills":
                print_skills(SKILLS)
                continue
            if user_input == "/jobs":
                print(list_nanocode_jobs())
                continue
            if user_input == "/tools":
                for name in selected_tool_specs(tool_names):
                    print(f"{GREEN}{name}{RESET}")
                continue
            if user_input.startswith("/mcp"):
                parts = user_input.split(maxsplit=2)
                sub = parts[1] if len(parts) > 1 else "status"
                if sub == "status":
                    print(mcp_status())
                elif sub == "search":
                    query = parts[2] if len(parts) > 2 else ""
                    print(mcp_search({"query": query, "refresh": True}))
                elif sub == "reload":
                    close_mcp_clients()
                    MCP_INVENTORY_CACHE = {"key": None, "inventory": None}
                    print(f"{GREEN}⏺ MCP reloaded{RESET}")
                else:
                    print("Usage: /mcp [status|search <query>|reload]")
                continue
            if user_input.startswith("/btw"):
                answer = run_btw(user_input[len("/btw") :])
                print(f"\n{YELLOW}BTW sidecar result{RESET}\n{DIM}Not added to session context.{RESET}\n\n{answer}")
                continue
            if user_input.startswith("/skill:"):
                raw = user_input[len("/skill:") :]
                name, _, skill_args = raw.partition(" ")
                invocation, error = skill_invocation_message(name, skill_args, SKILLS)
                if error:
                    print(f"{RED}⏺ {error}{RESET}")
                    continue
                user_input = invocation

            # Normal user messages start a worker thread so the prompt can keep
            # accepting native steering/follow-up input.
            drain_nanocode_job_events(messages)
            start_native_agent(user_input)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


# =============================================================================
# 12. CLI argument parsing and entrypoint
# =============================================================================

def parse_cli(argv):
    parser = argparse.ArgumentParser(description="nanocode - minimal OpenAI-compatible coding agent")
    parser.add_argument("message", nargs="*", help="Prompt for one-shot mode")
    parser.add_argument("-p", "--print", dest="print_mode", action="store_true", help="Run one prompt and print the result")
    parser.add_argument("--mode", choices=("text", "json"), default="text", help="Output mode for one-shot runs")
    parser.add_argument("--no-session", action="store_true", help="Accepted for headless compatibility; sessions are not persisted")
    parser.add_argument("--tools", help="Advanced override: none, all/default, or comma-separated tool names. Omitted uses the unified tool set.")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--system-prompt", action="append", default=[], help="Replace default system prompt with text or a file path")
    parser.add_argument("--append-system-prompt", action="append", default=[], help="Append system prompt text or a file path")
    parser.add_argument("--no-context-files", "-nc", action="store_true", help="Disable .nanocode/AGENTS.md / .nanocode/CLAUDE.md context discovery")
    parser.add_argument("--quiet", action="store_true", help="Suppress streaming/tool display in one-shot mode")
    return parser.parse_args(argv)


def run_cli_once(args):
    global MODEL
    if args.model:
        MODEL = args.model
    if args.quiet or args.mode == "json":
        os.environ["NANOCODE_QUIET_TOOLS"] = "1"

    prompt = " ".join(args.message).strip()
    if not prompt:
        raise SystemExit("one-shot mode requires a prompt")

    tool_names = parse_tool_names(args.tools, None)
    system_override = None
    if args.system_prompt:
        system_override = "\n\n".join(resolve_prompt_arg(item).strip() for item in args.system_prompt if resolve_prompt_arg(item).strip())
    skills = load_skills(os.getcwd())
    system_prompt = build_system_prompt(
        os.getcwd(),
        skills,
        tool_names=tool_names,
        system_prompt_override=system_override,
        append_system_prompts=args.append_system_prompt,
        include_context=not args.no_context_files,
    )
    display = not args.quiet and args.mode != "json"

    def emit_event(event):
        if args.mode == "json":
            print(json.dumps(event, ensure_ascii=False), flush=True)

    result = run_agent_once(
        prompt,
        system_prompt,
        tool_names=tool_names,
        display=display,
        event_callback=emit_event if args.mode == "json" else None,
    )

    if args.mode == "json":
        emit_event({"type": "agent_end", "text": result["text"], "messages": result["messages"]})
    elif args.quiet:
        print(result["text"])


def cli(argv=None):
    args = parse_cli(argv if argv is not None else sys.argv[1:])
    if args.print_mode or args.mode == "json" or args.message:
        run_cli_once(args)
    else:
        main(parse_tool_names(args.tools, None))


if __name__ == "__main__":
    cli()

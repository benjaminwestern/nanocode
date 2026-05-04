# nanocode

A single-file, zero-dependency OpenAI-compatible coding agent.

Nanocode defaults to [OpenCode Go](https://dev.opencode.ai/docs/go/) via Chat Completions, streams model output, exposes a unified tool surface, and keeps its harness config isolated under `~/.nanocode` and project `.nanocode/`.

![screenshot](screenshot.png)

## Features

### OpenAI-compatible agent loop

- Uses an OpenAI-compatible `/chat/completions` endpoint.
- Defaults to OpenCode Go at `https://opencode.ai/zen/go/v1` with model `kimi-k2.6`.
- Reuses OpenCode auth from `~/.local/share/opencode/auth.json` when available.
- Supports `OPENCODE_API_KEY`, `OPENCODE_GO_API_KEY`, or `OPENAI_API_KEY`.
- Streams assistant text, reasoning/thinking, and tool calls as they arrive.

### Unified tool surface

Nanocode exposes one default tool set in interactive and headless modes:

- Pi-compatible local tools: `read`, `write`, `edit`, `bash`
- File helpers: `glob`, `find`, `ls`, `grep`
- Web tools: `websearch`, `webfetch`
- MCP tools: `mcp_search`, `mcp_inspect`, `mcp_call`, plus configured direct `mcp__server__tool` surfaces
- Self-launch tool: `nanocode`

`--tools` is only an advanced override for constrained smoke/headless runs. Omitted `--tools` uses the unified surface.

### Pi-compatible local tools

| Tool | Description |
|------|-------------|
| `read` | Read file contents with `path`, 1-indexed `offset`, and `limit`; output is truncated for large files |
| `write` | Create or overwrite a file, creating parent directories |
| `edit` | Batch exact replacements via `edits: [{oldText, newText}]`; old text must be unique and non-overlapping |
| `bash` | Run a bash command with `command` and optional `timeout`; output streams and mise is refreshed first |

### Web and MCP

- `webfetch` fetches HTTP(S) URLs as markdown, text, or HTML.
- `websearch` uses Exa when configured and falls back to DuckDuckGo HTML search.
- MCP config is read from `~/.nanocode/mcp.json` plus ancestor `.nanocode/mcp.json` files.
- Direct MCP tool surfaces are exposed as `mcp__server__tool` when configured.

### Context files and skills

Nanocode loads local harness context from:

- `~/.nanocode/AGENTS.md`
- project `.nanocode/AGENTS.md` / `.nanocode/agents.md`
- project `.nanocode/CLAUDE.md` / `.nanocode/claude.md`
- nearest `.nanocode/SYSTEM.md` / `.nanocode/system.md`, falling back to `~/.nanocode/SYSTEM.md`
- `~/.nanocode/APPEND_SYSTEM.md` and project `.nanocode/APPEND_SYSTEM.md` / `.nanocode/append_system.md`

Skill discovery supports:

- ancestor `.nanocode/skills`
- `~/.nanocode/skills`
- shared `~/.agents/skills`

Nanocode does **not** read Pi’s `.pi` config paths.

### Headless child jobs

The `nanocode` tool can launch isolated child nanocode processes.

- `action="run"` runs synchronously and returns a compact result.
- `action="start"` starts a background job.
- Background jobs support `list`, `status`, `read`, and `cancel`.
- Completed background jobs auto-emit their result into the parent session at the next safe boundary.

### Native steering and follow-up queue

While the parent agent is working, the interactive prompt remains available.

- Plain input while working queues a steering message.
- `/steer <message>` queues steering for the next safe boundary before the next model call.
- `/queue <message>` or `/followup <message>` queues a follow-up after current work finishes.
- `/queue` lists queued messages.
- `/dequeue` clears queued messages and prints them for re-entry.
- `/dequeue edit` edits queued messages as JSON in `$VISUAL` or `$EDITOR`.

This queue is native to the parent process only; child nanocode processes do not get their own interactive queue.

### Context compaction

Nanocode estimates context usage locally and compacts older turns before a request when remaining context falls below a reserve.

Defaults:

- `NANOCODE_CONTEXT_WINDOW_TOKENS=128000`
- `NANOCODE_COMPACTION_RESERVE_TOKENS=16384`
- `NANOCODE_COMPACTION_REMAINING_RATIO=0.15`
- `NANOCODE_COMPACTION_KEEP_RECENT_TOKENS=20000`

Disable with:

```bash
export NANOCODE_COMPACTION=0
```

Manual controls:

- `/context` shows current estimated usage.
- `/compact [instructions]` summarizes older conversation history and keeps recent work.

### Sidecar questions

`/btw` asks a throw-away sidecar question without adding it to the parent conversation history.

Use `/btw --tools <question>` to allow the sidecar to use the unified tool surface while instructed to stay non-mutating.

## Usage

```bash
python nanocode.py
```

Set a model:

```bash
MODEL="deepseek-v4-pro" python nanocode.py
```

Or switch inside the REPL:

```text
/model deepseek-v4-pro
```

Use a custom OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:4010/v1"
export OPENAI_API_KEY="unused"
export MODEL="your-model-id"
python nanocode.py
```

Hide streamed thinking:

```bash
export SHOW_THINKING=0
```

## One-shot / headless mode

```bash
python nanocode.py --print "summarise this repo"
python nanocode.py --print --mode json --quiet "find auth code"
```

Useful flags:

- `--mode json` emits JSON events for streaming child progress.
- `--quiet` suppresses terminal rendering in one-shot mode.
- `--model <id>` overrides the model for the run.
- `--system-prompt <text-or-file>` replaces the default system prompt.
- `--append-system-prompt <text-or-file>` appends extra system instructions.
- `--no-context-files` disables `.nanocode` context discovery.
- `--tools none` or `--tools read,bash` constrains tools for explicit smoke/headless runs.

## Commands

| Command | Description |
|---------|-------------|
| `/c` | Clear conversation |
| `/compact [instructions]` | Summarise older conversation history and keep recent work |
| `/context` | Show estimated context usage and auto-compaction threshold |
| `/model` | Show current model and available `/models` results |
| `/model <id-or-search>` | Switch to an exact model id, show matches, or set the id directly if the endpoint cannot verify it |
| `/reload` | Reload skills, context files, jobs, and MCP config |
| `/skills` | List discovered skills |
| `/skill:<name> [args]` | Load a skill and run it with optional args |
| `/jobs` | List headless nanocode background jobs |
| `/queue` | List queued native steering/follow-up messages |
| `/queue <message>` / `/followup <message>` | Queue a native follow-up |
| `/steer <message>` | Queue native steering before the next model call |
| `/dequeue` | Clear queued native messages and print them |
| `/dequeue edit` | Edit queued native messages in `$VISUAL` / `$EDITOR` |
| `/tools` | List active nanocode tools |
| `/mcp status` | Show enabled MCP servers |
| `/mcp search <query>` | Search MCP tool inventory |
| `/mcp reload` | Clear MCP cache and restart MCP stdio clients |
| `/btw <question>` | Ask a throw-away sidecar question outside parent history |
| `/btw --tools <question>` | Sidecar with unified tools enabled and non-mutating instructions |
| `/q` / `exit` | Quit |

## Tool reference

| Tool | Description |
|------|-------------|
| `read` | Read file contents with offset/limit |
| `write` | Create or overwrite files |
| `edit` | Apply exact batch replacements |
| `bash` | Run bash commands with optional timeout |
| `glob` | Find files by glob pattern, sorted by mtime |
| `find` | Find files by pattern under a path |
| `ls` | List files and directories |
| `grep` | Search files for regex |
| `nanocode` | Run isolated child nanocode processes or manage background jobs |
| `websearch` | Search current web content |
| `webfetch` | Fetch HTTP(S) URLs |
| `mcp_search` | Search enabled MCP tool inventory |
| `mcp_inspect` | Inspect an MCP tool schema |
| `mcp_call` | Call an MCP tool after inspection |

## Configuration paths

| Purpose | Paths |
|---------|-------|
| Global nanocode dir | `~/.nanocode` |
| Project nanocode dir | ancestor `.nanocode/` directories |
| MCP | `~/.nanocode/mcp.json`, `.nanocode/mcp.json` |
| Context | `AGENTS.md`, `agents.md`, `CLAUDE.md`, `claude.md` under nanocode dirs |
| System prompt | `.nanocode/SYSTEM.md`, `.nanocode/system.md`, `~/.nanocode/SYSTEM.md` |
| Append prompt | `APPEND_SYSTEM.md`, `append_system.md`, `append-system.md` under nanocode dirs |
| Skills | `.nanocode/skills`, `~/.nanocode/skills`, `~/.agents/skills` |

## Example

```text
────────────────────────────────────────
❯ what files are here?
────────────────────────────────────────

⏺ bash(command=ls -1)
  ⎿  README.md ... +2 lines

⏺ This directory contains `README.md`, `nanocode.py`, and `screenshot.png`.
```

## License

MIT

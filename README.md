# ccr-py

A lightweight proxy server that accepts [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) requests and forwards them to any OpenAI-compatible backend, converting formats in both directions. The client sees a standard Anthropic API; the backend sees a standard OpenAI API.

```
Claude Code / any Anthropic client
        |  Anthropic format
        v
    ccr-py (this server)
        |  OpenAI format
        v
  OpenAI-compatible provider
```

## Quick Start

```bash
pip install fastapi uvicorn gunicorn httpx
```

Point your Anthropic client at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:3456
export ANTHROPIC_API_KEY=any-value   # validated by the backend, not by ccr-py
```

## Running the server

### Development (single process)

```bash
python main.py --config config.json
```

### Production (recommended)

```bash
gunicorn server:app \
  -w 32 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:3456 \
  --timeout 850 \
  --graceful-timeout 10 \
  --preload \
  -e CCR_CONFIG=config.json
```

**Key flags:**

| Flag | Value | Reason |
|---|---|---|
| `-w 32` | worker count | IO-bound workload; ~2× CPU is a good baseline. Adjust based on memory and concurrency needs. |
| `-k uvicorn.workers.UvicornWorker` | worker class | Async event loop per worker; handles many concurrent streaming responses efficiently. |
| `--timeout 850` | seconds | Should exceed `API_TIMEOUT_MS / 1000` (120 s default) plus any streaming duration. Set higher than your longest expected response. |
| `--graceful-timeout 10` | seconds | Time for in-flight requests to finish before a worker is killed on reload/shutdown. |
| `--preload` | — | Master process imports the app before forking workers — saves memory via copy-on-write and catches import errors early. |
| `-e CCR_CONFIG=config.json` | env var | Each worker reads this to load the config file independently after fork. |

Alternatively via `main.py`:

```bash
python main.py --config config.json --workers 32
```

### Workers guideline

Each worker is an independent process with its own asyncio event loop. A single worker already handles many concurrent streaming connections. Rule of thumb:

- Start at `2 × nproc` for IO-bound workloads
- Reduce if memory is tight (~50 MB per worker)
- `1` worker is sufficient for low-traffic or development use

## Configuration

`config.json` example:

```json
{
  "PORT": 3456,
  "API_TIMEOUT_MS": 120000,
  "Providers": [
    {
      "name": "my-provider",
      "api_base_url": "http://your-openai-endpoint/v1/chat/completions",
      "api_key": "$MY_API_KEY",
      "max_retries": 3,
      "params": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "reasoning": {"budget_tokens": 8000}
      }
    }
  ],
  "Router": {
    "default": "my-provider,model-name"
  }
}
```

### Top-level fields

| Field | Default | Description |
|---|---|---|
| `PORT` | `3456` | Listen port |
| `HOST` | `0.0.0.0` | Listen host |
| `API_TIMEOUT_MS` | `600000` | Request timeout in milliseconds |
| `LOG_LEVEL` | `info` | Log level: `debug` / `info` / `warning` / `error` |

### Provider fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Identifier used in `Router` |
| `api_base_url` | yes | Full URL of the OpenAI-compatible endpoint (e.g. `.../v1/chat/completions`) |
| `api_key` | yes | API key sent as `Authorization: Bearer <key>` |
| `max_retries` | no | Retries on 429/5xx (default `3`) |
| `params` | no | Parameter defaults/overrides (see below) |

### `params` — provider-level parameter overrides

All fields are optional. Applied to every request sent to this provider.

| Field | Description |
|---|---|
| `temperature` | Default temperature if the request does not specify one |
| `top_p` | Default top_p if the request does not specify one |
| `max_tokens` | Default max_tokens; also acts as a ceiling if the request specifies a higher value |
| `reasoning` | Object `{"budget_tokens": N}` — enables extended thinking if the request does not already include a `thinking` config |

### Router

```json
"Router": {
  "default": "provider-name,model-name"
}
```

The model name is injected into every outgoing request and echoed back in the Anthropic response. Only `default` is used today.

### Environment variable interpolation

Any string value in `config.json` supports `$VAR` or `${VAR}` substitution:

```json
{ "api_key": "$MY_API_KEY" }
```

---

## Request field conversion (Anthropic → OpenAI)

### Top-level sampling params

| Anthropic field | OpenAI field | Notes |
|---|---|---|
| `model` | `model` | Overridden by the router's configured model |
| `max_tokens` | `max_tokens` | Also capped by provider `params.max_tokens` |
| `temperature` | `temperature` | Provider default applied if absent |
| `top_p` | `top_p` | Provider default applied if absent |
| `top_k` | `top_k` | Passed through; ignored by providers that don't support it |
| `stop_sequences` | `stop` | Direct rename |
| `stream` | `stream` | When true, also adds `stream_options: {include_usage: true}` |

### Thinking / extended reasoning

| Anthropic `thinking.type` | Behaviour |
|---|---|
| `enabled` | Forwarded as-is with `budget_tokens`. Provider must support it. |
| `adaptive` | Forwarded as-is. No `budget_tokens` required; provider decides budget. |
| `disabled` | `thinking` field omitted from outgoing request. |

Provider-level default via `params.reasoning`:
- If the request has no `thinking` and `params.reasoning` is set, injects `{"type": "enabled", "budget_tokens": N}`.

### Tool choice

| Anthropic `tool_choice.type` | OpenAI `tool_choice` |
|---|---|
| `auto` | `"auto"` |
| `any` | `"required"` |
| `tool` | `{"type": "function", "function": {"name": "..."}}` |
| `none` | `"none"` |

`disable_parallel_tool_use: true` on any variant → `parallel_tool_calls: false`

### Tools

| Anthropic field | OpenAI field |
|---|---|
| `tool.name` | `function.name` |
| `tool.description` | `function.description` |
| `tool.input_schema` | `function.parameters` |
| `tool.strict` | `function.strict` |

### `output_config`

| Anthropic field | OpenAI field | Notes |
|---|---|---|
| `output_config.effort` | `reasoning_effort` | `low/medium/high` passed through; `max` → `xhigh` |
| `output_config.format` (json_schema) | `response_format` | `{type: "json_schema", json_schema: <schema>}` |

### `metadata`

| Anthropic field | OpenAI field |
|---|---|
| `metadata.user_id` | `user` |

### Messages and content blocks

| Anthropic content block | Converted to |
|---|---|
| `text` (user) | `{type: "text", text: "..."}` part |
| `image` (base64) | `{type: "image_url", image_url: {url: "data:<mt>;base64,..."}}` |
| `image` (URL) | `{type: "image_url", image_url: {url: "..."}}` |
| `tool_result` | OpenAI `role: "tool"` message with `tool_call_id` |
| `text` (assistant) | `content` string |
| `tool_use` (assistant) | `tool_calls` array |
| `thinking` (assistant history) | Skipped — not sent back to provider |

### Not converted (Anthropic-only)

These fields are specific to the Anthropic platform and have no OpenAI equivalent:

| Field | Reason |
|---|---|
| `cache_control` | Anthropic prompt caching — no OpenAI equivalent |
| `container` | Anthropic code execution containers |
| `inference_geo` | Anthropic data-residency routing |
| `service_tier` | Anthropic billing tier |
| `tool.cache_control` | Anthropic prompt caching on tools |
| `system` content block `cache_control` | Same |

---

## Response field conversion (OpenAI → Anthropic)

### Non-streaming

| OpenAI field | Anthropic field | Notes |
|---|---|---|
| `id` | `id` | |
| `choices[0].message.content` | `content[].text` block | |
| `choices[0].message.tool_calls` | `content[].tool_use` blocks | `arguments` string parsed to JSON |
| `choices[0].message.thinking.content` | `content[].thinking` block | OpenAI native thinking format |
| `choices[0].message.reasoning_content` | `content[].thinking` block | Common third-party provider format |
| `choices[0].finish_reason` | `stop_reason` | See mapping below |
| `usage.prompt_tokens` | `usage.input_tokens` | Minus cache tokens |
| `usage.completion_tokens` | `usage.output_tokens` | |
| `usage.prompt_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` | |
| `usage.prompt_tokens_details.cache_creation_tokens` | `usage.cache_creation_input_tokens` | |

### `finish_reason` → `stop_reason` mapping

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `stop_sequence` |
| _(anything else)_ | `end_turn` |

### Streaming SSE event sequence

```
message_start
ping
  [for each content block:]
  content_block_start
    content_block_delta  (repeated)
  content_block_stop
message_delta          ← stop_reason + final usage
message_stop
```

Supported delta types:

| Source | Anthropic delta type |
|---|---|
| `delta.content` (text) | `text_delta` |
| `delta.thinking.content` | `thinking_delta` |
| `delta.thinking.signature` | `signature_delta` |
| `delta.reasoning_content` | `thinking_delta` |
| `delta.tool_calls[].function.arguments` | `input_json_delta` |

---

## CLI

```
python main.py [--config PATH] [--host HOST] [--port PORT] [--log-level LEVEL]
```

CLI flags override the corresponding `config.json` values.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Main Messages API — streaming and non-streaming |
| `GET` | `/health` | Returns `{"status": "ok"}` |

"""
FastAPI server — accepts Anthropic /v1/messages requests and forwards them
to an OpenAI-compatible provider, converting formats in both directions.
"""

import logging
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from client import ProviderError, post_json, stream_lines
from config import apply_provider_params, get_provider, resolve_route
from converter import anthropic_to_openai, openai_to_anthropic, stream_openai_to_anthropic

logger = logging.getLogger(__name__)

app = FastAPI(title="Claude Code Router (Python)")

# Populated by main.py before server starts
_config: dict = {}


def set_config(cfg: dict) -> None:
    global _config
    _config = cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_headers(provider: dict) -> dict:
    api_key = provider.get("api_key", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_provider(anthropic_req: dict):
    """Determine which provider + model to use and return (provider, model, url)."""
    route = resolve_route(_config)
    if route is None:
        raise HTTPException(500, "No default route configured")

    provider_name, model = route
    provider = get_provider(_config, provider_name)
    if provider is None:
        raise HTTPException(500, f"Provider '{provider_name}' not found in config")

    return provider, model, provider["api_base_url"]


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    try:
        provider, model, url = _get_provider(body)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Routing error")
        raise HTTPException(500, str(exc))

    # Override model in the original request so converter uses the routed model
    body = dict(body)
    body["model"] = model

    # Anthropic → OpenAI, then apply provider param defaults
    openai_req = anthropic_to_openai(body)
    openai_req = apply_provider_params(provider, openai_req)

    headers = _provider_headers(provider)
    max_retries: int = provider.get("max_retries", 3)
    timeout: float = _config.get("API_TIMEOUT_MS", 600_000) / 1000

    is_stream = openai_req.get("stream", False)

    if is_stream:
        return StreamingResponse(
            _stream_response(openai_req, url, headers, model, max_retries, timeout),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    try:
        openai_resp = await post_json(
            url, headers, openai_req, timeout=timeout, max_retries=max_retries
        )
    except ProviderError as exc:
        logger.error("Provider error: %s", exc)
        raise HTTPException(exc.status or 502, exc.body or str(exc))
    except Exception as exc:
        logger.exception("Unexpected error calling provider")
        raise HTTPException(502, str(exc))

    anthropic_resp = openai_to_anthropic(openai_resp, model)
    return anthropic_resp


async def _stream_response(
    openai_req: dict,
    url: str,
    headers: dict,
    model: str,
    max_retries: int,
    timeout: float,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    try:
        lines = stream_lines(url, headers, openai_req, timeout=timeout, max_retries=max_retries)
        async for event in stream_openai_to_anthropic(lines, message_id, model):
            yield event
    except ProviderError as exc:
        # Emit an Anthropic-format error event
        import json
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": exc.body or str(exc)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
    except Exception as exc:
        import json
        logger.exception("Streaming error")
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(exc)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

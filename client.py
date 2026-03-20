"""Async HTTP client with retry logic for forwarding requests to providers."""

import asyncio
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}

# Shared HTTP client for non-streaming requests (connection pooling)
_shared_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    """Return the shared HTTP client (connection pooling). Created lazily."""
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0), trust_env=False)
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared client. Called during app shutdown."""
    global _shared_client
    if _shared_client:
        await _shared_client.aclose()
        _shared_client = None


class ProviderError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"Provider returned HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


class ProviderStream:
    """Async iterable yielding SSE lines from an established provider connection."""

    def __init__(self, response: httpx.Response, client: httpx.AsyncClient):
        self._response = response
        self._client = client

    async def __aiter__(self):
        async for line in self._response.aiter_lines():
            if line:
                yield line.encode()

    async def aclose(self):
        await self._response.aclose()
        await self._client.aclose()


async def post_json(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> dict:
    """POST JSON, return parsed response dict. Retries on transient errors."""
    last_exc: Exception | None = None
    client = get_shared_client()

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning("Retry %d/%d after sleep", attempt, max_retries)
            await asyncio.sleep(1)

        try:
            resp = await client.post(url, headers=headers, json=body, timeout=timeout)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            logger.warning("Connection error (attempt %d): %s", attempt + 1, exc)
            continue

        if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
            logger.warning(
                "HTTP %d from provider (attempt %d), will retry",
                resp.status_code, attempt + 1,
            )
            last_exc = ProviderError(resp.status_code, resp.text)
            continue

        if resp.status_code >= 400:
            raise ProviderError(resp.status_code, resp.text)

        return resp.json()

    raise last_exc or ProviderError(0, "Unknown error after retries")


async def open_provider_stream(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> ProviderStream:
    """
    Eagerly open a streaming connection to the provider.
    Returns a ProviderStream (async iterable of SSE lines).
    Raises ProviderError immediately if provider returns an error status.
    Retries on transient errors before raising.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning("Stream retry %d/%d after sleep", attempt, max_retries)
            await asyncio.sleep(1)

        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), trust_env=False)
        try:
            req = client.build_request("POST", url, headers=headers, json=body)
            resp = await client.send(req, stream=True)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            logger.warning("Stream connection error (attempt %d): %s", attempt + 1, exc)
            await client.aclose()
            continue

        if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
            logger.warning(
                "HTTP %d from provider (stream attempt %d), will retry",
                resp.status_code, attempt + 1,
            )
            last_exc = ProviderError(resp.status_code, "")
            await resp.aclose()
            await client.aclose()
            continue

        if resp.status_code >= 400:
            body_text = await resp.aread()
            await resp.aclose()
            await client.aclose()
            raise ProviderError(resp.status_code, body_text.decode())

        return ProviderStream(resp, client)

    raise last_exc or ProviderError(0, "Unknown error after retries")


async def stream_lines(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> AsyncIterator[bytes]:
    """POST and yield raw SSE lines. Retries on transient errors."""
    stream = await open_provider_stream(url, headers, body, timeout, max_retries)
    try:
        async for line in stream:
            yield line
    finally:
        await stream.aclose()

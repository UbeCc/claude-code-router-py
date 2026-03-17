import asyncio
import logging
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class ProviderError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"Provider returned HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


async def post_json(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> dict:
    """POST JSON, return parsed response dict. Retries on transient errors."""
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning("Retry %d/%d after sleep", attempt, max_retries)
                await asyncio.sleep(1)

            try:
                resp = await client.post(url, headers=headers, json=body)
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


async def stream_lines(
    url: str,
    headers: dict,
    body: dict,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> AsyncIterator[bytes]:
    """
    POST and yield raw SSE lines.  Retries the *connection* on transient
    errors; once data starts flowing retries are not possible.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning("Stream retry %d/%d after sleep", attempt, max_retries)
            await asyncio.sleep(1)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code in _RETRY_STATUSES and attempt < max_retries:
                        logger.warning(
                            "HTTP %d from provider (stream attempt %d), will retry",
                            resp.status_code, attempt + 1,
                        )
                        last_exc = ProviderError(resp.status_code, "")
                        continue

                    if resp.status_code >= 400:
                        body_text = await resp.aread()
                        raise ProviderError(resp.status_code, body_text.decode())

                    async for line in resp.aiter_lines():
                        if line:
                            yield line.encode()
            return  # success

        except ProviderError:
            raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            logger.warning("Stream connection error (attempt %d): %s", attempt + 1, exc)
            continue

    raise last_exc or ProviderError(0, "Unknown error after retries")
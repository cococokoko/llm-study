"""
client.py — Async OpenRouter API client with retry, rate-limiting, and
            full response provenance capture.

Uses the openai-compatible REST interface:
  POST https://openrouter.ai/api/v1/chat/completions
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Errors worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    response_text: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    finish_reason: Optional[str]
    latency_ms: int
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """
    Thin async wrapper around the OpenRouter chat completions endpoint.

    Parameters
    ----------
    api_key      : OpenRouter API key (or set OPENROUTER_API_KEY env var)
    http_referer : Optional HTTP-Referer header (shown in OR dashboard)
    site_name    : Optional X-Title header
    timeout      : Per-request timeout in seconds
    max_retries  : Maximum retry attempts on transient errors
    """

    def __init__(
        self,
        api_key: str | None = None,
        http_referer: str = "https://github.com/longitudinal-llm-study",
        site_name: str = "LLM Longitudinal Study",
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._referer = http_referer
        self._site_name = site_name
        self._timeout = timeout
        self._max_retries = max_retries

        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE,
            headers=self._base_headers(),
            timeout=httpx.Timeout(timeout),
        )

    def _base_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._referer,
            "X-Title": self._site_name,
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a single chat completion request and return an LLMResponse.

        On transient failures the request is retried with exponential back-off.
        On permanent failure (or exhausted retries) the error is captured in
        LLMResponse.error without raising, so the pipeline can continue.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if extra_params:
            body.update(extra_params)

        t0 = time.monotonic()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                retry=retry_if_exception_type(
                    (httpx.TransportError, _RetryableHTTPError)
                ),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post(
                        "/chat/completions", content=json.dumps(body)
                    )
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise _RetryableHTTPError(resp.status_code, resp.text)
                    resp.raise_for_status()

            latency_ms = int((time.monotonic() - t0) * 1000)
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                response_text=choice["message"]["content"],
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                finish_reason=choice.get("finish_reason"),
                latency_ms=latency_ms,
                raw=data,
            )

        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return LLMResponse(
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RetryableHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status


# ---------------------------------------------------------------------------
# Convenience: fetch available models from OpenRouter
# ---------------------------------------------------------------------------

async def list_available_models(api_key: str | None = None) -> list[dict[str, Any]]:
    """Return the OpenRouter model catalogue (id, name, context_length, pricing)."""
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    async with httpx.AsyncClient(timeout=30) as hc:
        resp = await hc.get(
            f"{OPENROUTER_BASE}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

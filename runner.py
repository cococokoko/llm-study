"""
runner.py — Async batch runner for the LLM longitudinal study pipeline.

Orchestration model
───────────────────
  For each wave, iterate over (prompt × model) Cartesian product.
  Already-completed pairs (error IS NULL) are skipped automatically.
  Concurrency is bounded by a semaphore; a per-second token bucket
  enforces the user-specified rate limit across all concurrent workers.

  Progress is reported live via the `rich` library.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from string import Formatter
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from client import OpenRouterClient
from db import (
    fetch_responses,
    list_models,
    list_prompts,
    pending_jobs,
    save_response,
)

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple async token bucket for requests-per-second control."""

    def __init__(self, rps: float) -> None:
        self._rps = rps
        self._tokens = rps
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rps, self._tokens + elapsed * self._rps)
            self._last = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rps
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def _render(template: str, variables: dict[str, Any]) -> str:
    """
    Safe Python str.format_map substitution.
    Missing keys are left as-is (e.g., {missing} stays verbatim).
    """
    class _SafeMap(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_SafeMap(variables))


# ---------------------------------------------------------------------------
# Single-job worker
# ---------------------------------------------------------------------------

@dataclass
class _Job:
    wave_id: str
    prompt: sqlite3.Row
    model: sqlite3.Row
    extra_vars: dict[str, Any]


async def _run_job(
    job: _Job,
    conn: sqlite3.Connection,
    llm: OpenRouterClient,
    bucket: _TokenBucket,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: TaskID,
    conn_lock: asyncio.Lock,
) -> None:
    """Execute one (prompt, model) pair and persist the result."""
    async with semaphore:
        await bucket.acquire()

        p = job.prompt
        m = job.model

        # Merge default variables with any extra vars passed at runtime
        defaults: dict[str, Any] = json.loads(p["variables"] or "{}")
        defaults.update(job.extra_vars)

        prompt_rendered = _render(p["template"], defaults)
        system_rendered = _render(p["system_msg"], defaults) if p["system_msg"] else None

        params: dict[str, Any] = json.loads(m["parameters"] or "{}")
        temperature   = params.pop("temperature", 0.7)
        max_tokens    = params.pop("max_tokens", 1024)
        top_p         = params.pop("top_p", 1.0)

        result = await llm.chat(
            model=m["model_id"],
            prompt=prompt_rendered,
            system=system_rendered,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_params=params or None,
        )

        async with conn_lock:
            save_response(
                conn,
                wave_id=job.wave_id,
                prompt_id=p["id"],
                model_config_id=m["id"],
                prompt_rendered=prompt_rendered,
                system_rendered=system_rendered,
                response_text=result.response_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                finish_reason=result.finish_reason,
                latency_ms=result.latency_ms,
                error=result.error,
            )

        status = "[red]ERR[/]" if result.error else "[green]OK[/]"
        progress.advance(task_id)
        progress.print(
            f"  {status} {m['display_name']:40s} | {p['label'][:40]:40s} | "
            f"{result.latency_ms:>5}ms"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_wave(
    conn: sqlite3.Connection,
    wave_id: str,
    *,
    api_key: str | None = None,
    concurrency: int = 5,
    requests_per_second: float = 3.0,
    extra_vars: dict[str, Any] | None = None,
    http_referer: str = "https://github.com/longitudinal-llm-study",
    site_name: str = "LLM Longitudinal Study",
) -> dict[str, int]:
    """
    Run all pending (prompt × model) pairs for *wave_id*.

    Returns a summary dict: {"completed": N, "errors": N, "skipped": N}.
    """
    jobs_raw = pending_jobs(conn, wave_id)
    if not jobs_raw:
        console.print("[yellow]No pending jobs for this wave — all done.[/]")
        return {"completed": 0, "errors": 0, "skipped": 0}

    jobs = [
        _Job(
            wave_id=wave_id,
            prompt=p,
            model=m,
            extra_vars=extra_vars or {},
        )
        for p, m in jobs_raw
    ]

    console.rule(f"[bold blue]Wave  {wave_id[:8]}…  —  {len(jobs)} jobs")

    bucket    = _TokenBucket(requests_per_second)
    semaphore = asyncio.Semaphore(concurrency)
    conn_lock = asyncio.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("[cyan]Collecting responses", total=len(jobs))

    async with OpenRouterClient(
        api_key=api_key,
        http_referer=http_referer,
        site_name=site_name,
    ) as llm:
        with Live(progress, console=console, refresh_per_second=4):
            await asyncio.gather(
                *[
                    _run_job(j, conn, llm, bucket, semaphore, progress, task_id, conn_lock)
                    for j in jobs
                ]
            )

    # Tally results
    rows = fetch_responses(conn, wave_id)
    errors    = sum(1 for r in rows if r["error"] is not None)
    completed = sum(1 for r in rows if r["error"] is None)
    skipped   = 0  # already-complete pairs were excluded from jobs
    console.rule(
        f"[bold green]Done — {completed} OK | {errors} errors | {skipped} skipped"
    )
    return {"completed": completed, "errors": errors, "skipped": skipped}

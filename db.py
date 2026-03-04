"""
db.py — SQLite persistence layer for the LLM longitudinal study pipeline.

Schema overview
───────────────
  prompt_bank      : versioned prompt templates with metadata
  study_waves      : named time-points for longitudinal collection
  model_configs    : LLM endpoint + parameter configs
  response_records : every (wave × prompt × model) response, idempotent
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS prompt_bank (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL UNIQUE,
    category    TEXT,
    tags        TEXT DEFAULT '[]',   -- JSON array of strings
    system_msg  TEXT,                -- optional system prompt
    template    TEXT NOT NULL,       -- may contain {variable} placeholders
    variables   TEXT DEFAULT '{}',   -- JSON: default values for placeholders
    version     INTEGER DEFAULT 1,
    created_at  TEXT NOT NULL,
    active      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS study_waves (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}'    -- JSON: arbitrary study metadata
);

CREATE TABLE IF NOT EXISTS model_configs (
    id           TEXT PRIMARY KEY,
    model_id     TEXT NOT NULL,      -- OpenRouter model string e.g. openai/gpt-4o
    display_name TEXT,
    parameters   TEXT DEFAULT '{}',  -- JSON: temperature, max_tokens, top_p, …
    active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS response_records (
    id               TEXT PRIMARY KEY,
    wave_id          TEXT NOT NULL REFERENCES study_waves(id),
    prompt_id        TEXT NOT NULL REFERENCES prompt_bank(id),
    model_config_id  TEXT NOT NULL REFERENCES model_configs(id),
    prompt_rendered  TEXT NOT NULL,  -- fully rendered prompt sent to the API
    system_rendered  TEXT,           -- fully rendered system message
    response_text    TEXT,           -- null on error
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    finish_reason    TEXT,
    latency_ms       INTEGER,
    error            TEXT,           -- null on success
    created_at       TEXT NOT NULL,
    UNIQUE(wave_id, prompt_id, model_config_id)
);

CREATE INDEX IF NOT EXISTS idx_rr_wave   ON response_records(wave_id);
CREATE INDEX IF NOT EXISTS idx_rr_prompt ON response_records(prompt_id);
CREATE INDEX IF NOT EXISTS idx_rr_model  ON response_records(model_config_id);
"""

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (and initialise) the study database."""
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Prompt Bank
# ---------------------------------------------------------------------------

def upsert_prompt(
    conn: sqlite3.Connection,
    *,
    label: str,
    template: str,
    category: str = "",
    tags: list[str] | None = None,
    system_msg: str | None = None,
    variables: dict[str, Any] | None = None,
    version: int = 1,
) -> str:
    """Insert or replace a prompt; returns its id."""
    existing = conn.execute(
        "SELECT id FROM prompt_bank WHERE label = ?", (label,)
    ).fetchone()
    pid = existing["id"] if existing else _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO prompt_bank
                (id, label, category, tags, system_msg, template, variables,
                 version, created_at, active)
            VALUES (?,?,?,?,?,?,?,?,?,1)
            ON CONFLICT(label) DO UPDATE SET
                category   = excluded.category,
                tags       = excluded.tags,
                system_msg = excluded.system_msg,
                template   = excluded.template,
                variables  = excluded.variables,
                version    = excluded.version
            """,
            (
                pid, label, category,
                json.dumps(tags or []),
                system_msg,
                template,
                json.dumps(variables or {}),
                version,
                _now(),
            ),
        )
    return pid


def list_prompts(conn: sqlite3.Connection, active_only: bool = True) -> list[sqlite3.Row]:
    q = "SELECT * FROM prompt_bank"
    if active_only:
        q += " WHERE active = 1"
    return conn.execute(q).fetchall()


# ---------------------------------------------------------------------------
# Study Waves
# ---------------------------------------------------------------------------

def create_wave(
    conn: sqlite3.Connection,
    name: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a study wave; raises if name already exists."""
    wid = _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO study_waves (id, name, description, created_at, metadata)
            VALUES (?,?,?,?,?)
            """,
            (wid, name, description, _now(), json.dumps(metadata or {})),
        )
    return wid


def get_or_create_wave(
    conn: sqlite3.Connection,
    name: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    row = conn.execute(
        "SELECT id FROM study_waves WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    return create_wave(conn, name, description, metadata)


def list_waves(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM study_waves ORDER BY created_at"
    ).fetchall()


# ---------------------------------------------------------------------------
# Model Configs
# ---------------------------------------------------------------------------

def upsert_model(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    display_name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM model_configs WHERE model_id = ?", (model_id,)
    ).fetchone()
    mid = existing["id"] if existing else _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO model_configs (id, model_id, display_name, parameters, active)
            VALUES (?,?,?,?,1)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                parameters   = excluded.parameters
            """,
            (mid, model_id, display_name or model_id,
             json.dumps(parameters or {})),
        )
    return mid


def list_models(conn: sqlite3.Connection, active_only: bool = True) -> list[sqlite3.Row]:
    q = "SELECT * FROM model_configs"
    if active_only:
        q += " WHERE active = 1"
    return conn.execute(q).fetchall()


# ---------------------------------------------------------------------------
# Response Records
# ---------------------------------------------------------------------------

def pending_jobs(
    conn: sqlite3.Connection, wave_id: str
) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
    """
    Return (prompt_row, model_row) pairs that have not yet been completed
    for the given wave (i.e., missing from response_records or had an error).
    """
    prompts = list_prompts(conn)
    models  = list_models(conn)
    done = {
        (r["prompt_id"], r["model_config_id"])
        for r in conn.execute(
            "SELECT prompt_id, model_config_id FROM response_records "
            "WHERE wave_id = ? AND error IS NULL",
            (wave_id,),
        ).fetchall()
    }
    jobs = []
    for p in prompts:
        for m in models:
            if (p["id"], m["id"]) not in done:
                jobs.append((p, m))
    return jobs


def save_response(
    conn: sqlite3.Connection,
    *,
    wave_id: str,
    prompt_id: str,
    model_config_id: str,
    prompt_rendered: str,
    system_rendered: str | None = None,
    response_text: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    finish_reason: str | None = None,
    latency_ms: int | None = None,
    error: str | None = None,
) -> str:
    rid = _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO response_records
                (id, wave_id, prompt_id, model_config_id,
                 prompt_rendered, system_rendered,
                 response_text, input_tokens, output_tokens,
                 finish_reason, latency_ms, error, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(wave_id, prompt_id, model_config_id) DO UPDATE SET
                prompt_rendered = excluded.prompt_rendered,
                system_rendered = excluded.system_rendered,
                response_text   = excluded.response_text,
                input_tokens    = excluded.input_tokens,
                output_tokens   = excluded.output_tokens,
                finish_reason   = excluded.finish_reason,
                latency_ms      = excluded.latency_ms,
                error           = excluded.error,
                created_at      = excluded.created_at
            """,
            (
                rid, wave_id, prompt_id, model_config_id,
                prompt_rendered, system_rendered,
                response_text, input_tokens, output_tokens,
                finish_reason, latency_ms, error,
                _now(),
            ),
        )
    return rid


def fetch_responses(
    conn: sqlite3.Connection,
    wave_id: str | None = None,
) -> list[sqlite3.Row]:
    q = """
        SELECT
            rr.*,
            pb.label        AS prompt_label,
            pb.category     AS prompt_category,
            pb.tags         AS prompt_tags,
            mc.model_id     AS model_id,
            mc.display_name AS model_display_name,
            sw.name         AS wave_name
        FROM response_records rr
        JOIN prompt_bank  pb ON pb.id = rr.prompt_id
        JOIN model_configs mc ON mc.id = rr.model_config_id
        JOIN study_waves   sw ON sw.id = rr.wave_id
    """
    if wave_id:
        q += " WHERE rr.wave_id = ?"
        return conn.execute(q, (wave_id,)).fetchall()
    return conn.execute(q).fetchall()

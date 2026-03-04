#!/usr/bin/env python3
"""
pipeline.py — CLI entry point for the LLM longitudinal study pipeline.

Usage
─────
  # Run today's prompts across all configured models:
  python pipeline.py run

  # Export results to CSV:
  python pipeline.py export --format csv --out results/

  # Print a statistical report:
  python pipeline.py report
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

from analysis import (
    export_csv,
    export_daily_txt,
    export_json,
    export_jsonl,
    export_parquet,
    print_report,
    responses_to_df,
)
from db import (
    fetch_responses,
    get_or_create_wave,
    list_models,
    list_prompts,
    open_db,
    upsert_model,
    upsert_prompt,
)
from runner import run_wave

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_prompts_yaml(path: str = "prompts.yaml") -> list[dict]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data.get("prompts", [])


# ---------------------------------------------------------------------------
# Seed DB with models and prompts from YAML files
# ---------------------------------------------------------------------------

def seed_db(conn, cfg: dict) -> None:
    models = cfg.get("models", [])
    for m in models:
        upsert_model(
            conn,
            model_id=m["model_id"],
            display_name=m.get("display_name"),
            parameters=m.get("parameters", {}),
        )

    prompts = load_prompts_yaml()
    for p in prompts:
        upsert_prompt(
            conn,
            label=p["label"],
            template=p["template"],
            category=p.get("category", ""),
            tags=p.get("tags", []),
            system_msg=p.get("system_msg"),
            variables=p.get("variables", {}),
            version=p.get("version", 1),
        )
    console.print(
        f"[dim]Seeded {len(models)} models and {len(prompts)} prompts.[/]"
    )


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace, cfg: dict) -> None:
    """Seed DB, then run today's prompts across all models."""
    db_path = cfg["study"]["db_path"]
    conn = open_db(db_path)

    seed_db(conn, cfg)

    today = datetime.date.today().isoformat()  # e.g. "2026-03-03"
    wave_id = get_or_create_wave(conn, name=today, description="daily run")
    console.print(f"Running daily wave: [bold]{today}[/]")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]Set OPENROUTER_API_KEY in .env or environment.[/]")
        sys.exit(1)

    study = cfg["study"]
    asyncio.run(
        run_wave(
            conn,
            wave_id,
            api_key=api_key,
            concurrency=study.get("concurrency", 5),
            requests_per_second=study.get("requests_per_second", 3.0),
            http_referer=study.get("http_referer", ""),
            site_name=study.get("site_name", "LLM Study"),
        )
    )

    # Auto-export today's responses as individual TXT files
    rows = fetch_responses(conn, wave_id)
    if rows:
        df = responses_to_df(rows)
        export_daily_txt(df, today)

    conn.close()


def cmd_export(args: argparse.Namespace, cfg: dict) -> None:
    """Export responses to file."""
    db_path = cfg["study"]["db_path"]
    conn = open_db(db_path)

    rows = fetch_responses(conn)
    if not rows:
        console.print("[yellow]No responses found.[/]")
        conn.close()
        return

    df = responses_to_df(rows)
    out = Path(getattr(args, "out", "."))
    out.mkdir(parents=True, exist_ok=True)
    fmt = getattr(args, "format", "csv")

    dispatch = {
        "csv":     lambda: export_csv(df, out / "responses.csv"),
        "json":    lambda: export_json(df, out / "responses.json"),
        "jsonl":   lambda: export_jsonl(df, out / "responses.jsonl"),
        "parquet": lambda: export_parquet(df, out / "responses.parquet"),
    }
    fn = dispatch.get(fmt)
    if fn is None:
        console.print(f"[red]Unknown format '{fmt}'. Choose: csv, json, jsonl, parquet[/]")
        sys.exit(1)
    fn()
    conn.close()


def cmd_report(args: argparse.Namespace, cfg: dict) -> None:
    """Print statistical summary."""
    db_path = cfg["study"]["db_path"]
    conn = open_db(db_path)
    rows = fetch_responses(conn)
    if not rows:
        console.print("[yellow]No responses to report.[/]")
        conn.close()
        return
    df = responses_to_df(rows)
    print_report(df)
    conn.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="LLM Study Pipeline",
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # run
    sub.add_parser("run", help="Run today's prompts across all models")

    # export
    pe = sub.add_parser("export", help="Export responses to file")
    pe.add_argument("--format", default="csv",
                    choices=["csv", "json", "jsonl", "parquet"])
    pe.add_argument("--out", default="results/", help="Output directory")

    # report
    sub.add_parser("report", help="Print statistical summary")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    dispatch = {
        "run":    cmd_run,
        "export": cmd_export,
        "report": cmd_report,
    }
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()

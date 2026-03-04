"""
analysis.py — Export and basic statistical analysis of collected responses.

Exports
───────
  • CSV / JSON / JSONL   — flat row per response
  • Parquet              — columnar (requires pyarrow)
  • Markdown summary     — descriptive statistics printed to stdout

Metrics computed
────────────────
  • Response length (characters, words)
  • Input / output token counts
  • Latency statistics (mean, median, p95)
  • Error rate per model
  • Cross-wave delta tables (for longitudinal comparison)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# pandas is required; pyarrow is optional (for parquet)
try:
    import pandas as pd
except ImportError:
    print("Install pandas:  pip install pandas", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Build DataFrame from DB rows
# ---------------------------------------------------------------------------

def responses_to_df(rows: list[sqlite3.Row]) -> "pd.DataFrame":
    """Convert fetch_responses() rows into a tidy DataFrame."""
    records = []
    for r in rows:
        text = r["response_text"] or ""
        records.append(
            {
                # identifiers
                "response_id":      r["id"],
                "wave_id":          r["wave_id"],
                "wave_name":        r["wave_name"],
                "prompt_id":        r["prompt_id"],
                "prompt_label":     r["prompt_label"],
                "prompt_category":  r["prompt_category"],
                "prompt_tags":      r["prompt_tags"],
                "model_config_id":  r["model_config_id"],
                "model_id":         r["model_id"],
                "model_display":    r["model_display_name"],
                # content
                "prompt_rendered":  r["prompt_rendered"],
                "system_rendered":  r["system_rendered"],
                "response_text":    r["response_text"],
                # metrics
                "input_tokens":     r["input_tokens"],
                "output_tokens":    r["output_tokens"],
                "finish_reason":    r["finish_reason"],
                "latency_ms":       r["latency_ms"],
                "error":            r["error"],
                "created_at":       r["created_at"],
                # derived
                "char_count":       len(text),
                "word_count":       len(text.split()) if text else 0,
                "success":          r["error"] is None,
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_csv(df: "pd.DataFrame", path: str | Path) -> None:
    df.to_csv(path, index=False)
    print(f"CSV exported → {path}")


def export_json(df: "pd.DataFrame", path: str | Path) -> None:
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    print(f"JSON exported → {path}")


def export_jsonl(df: "pd.DataFrame", path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in df.to_dict(orient="records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"JSONL exported → {path}")


def export_parquet(df: "pd.DataFrame", path: str | Path) -> None:
    try:
        df.to_parquet(path, index=False)
        print(f"Parquet exported → {path}")
    except ImportError:
        print("pyarrow not installed; skipping parquet export.")


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def summary_stats(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Per-model descriptive statistics across all waves and prompts.

    Returns a DataFrame with one row per model.
    """
    numeric_cols = ["latency_ms", "output_tokens", "char_count", "word_count"]
    g = df[df["success"]].groupby("model_display")

    agg: dict[str, Any] = {}
    for col in numeric_cols:
        if col in df.columns:
            agg[f"{col}_mean"]   = g[col].mean().round(1)
            agg[f"{col}_median"] = g[col].median().round(1)
            agg[f"{col}_p95"]    = g[col].quantile(0.95).round(1)

    agg["n_responses"] = g["response_id"].count()
    agg["error_rate"]  = (
        df.groupby("model_display")["success"]
        .apply(lambda s: round(1 - s.mean(), 4))
    )

    return pd.DataFrame(agg).reset_index()


def cross_wave_comparison(df: "pd.DataFrame", metric: str = "word_count") -> "pd.DataFrame":
    """
    Pivot table: rows = prompts, columns = (wave × model), values = metric.
    Useful for longitudinal analysis — detect drift in model outputs over time.
    """
    success = df[df["success"]].copy()
    pivot = success.pivot_table(
        index="prompt_label",
        columns=["wave_name", "model_display"],
        values=metric,
        aggfunc="mean",
    )
    pivot.columns = [" | ".join(c) for c in pivot.columns]
    return pivot.reset_index()


def error_report(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return rows where error is not null, with key columns."""
    cols = ["wave_name", "prompt_label", "model_display", "error", "latency_ms", "created_at"]
    return df[~df["success"]][cols].copy()


# ---------------------------------------------------------------------------
# Daily TXT export
# ---------------------------------------------------------------------------

def export_daily_txt(df: "pd.DataFrame", date: str, base_dir: str | Path = "results") -> None:
    """
    Save each response as an individual .txt file under results/YYYY-MM-DD/.

    File name pattern: {prompt_label}__{model_display}.txt
    Each file contains a metadata header followed by the response text.
    """
    out = Path(base_dir) / date
    out.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        model_slug = row["model_display"].replace(" ", "_").replace("/", "-")
        filename = f"{row['prompt_label']}__{model_slug}.txt"
        content_lines = [
            f"Date        : {date}",
            f"Model       : {row['model_display']}",
            f"Prompt      : {row['prompt_label']}",
            f"Category    : {row['prompt_category']}",
            f"Tokens in   : {row['input_tokens']}",
            f"Tokens out  : {row['output_tokens']}",
            f"Latency ms  : {row['latency_ms']}",
            f"Finish      : {row['finish_reason']}",
            f"Error       : {row['error'] or 'none'}",
            "",
            "── Prompt ──────────────────────────────────────────────────────────",
            str(row["prompt_rendered"] or ""),
            "",
            "── Response ────────────────────────────────────────────────────────",
            str(row["response_text"] or "(no response)"),
        ]
        (out / filename).write_text("\n".join(content_lines), encoding="utf-8")

    print(f"Daily TXT export → {out}  ({len(df)} files)")


# ---------------------------------------------------------------------------
# Markdown / text report
# ---------------------------------------------------------------------------

def print_report(df: "pd.DataFrame") -> None:
    print("\n" + "=" * 70)
    print("  LLM LONGITUDINAL STUDY — RESPONSE SUMMARY")
    print("=" * 70)

    waves  = df["wave_name"].unique().tolist()
    models = df["model_display"].unique().tolist()
    prompts = df["prompt_label"].unique().tolist()

    print(f"\nWaves   : {len(waves)}")
    print(f"Models  : {len(models)}")
    print(f"Prompts : {len(prompts)}")
    print(f"Total   : {len(df)} response records")
    print(f"Success : {df['success'].sum()} ({df['success'].mean():.1%})")

    print("\n── Per-model summary ──────────────────────────────────────────────")
    stats = summary_stats(df)
    print(stats.to_string(index=False))

    errs = error_report(df)
    if not errs.empty:
        print(f"\n── Errors ({len(errs)}) ─────────────────────────────────────────────")
        print(errs.to_string(index=False))

    if len(waves) > 1:
        print("\n── Cross-wave word-count comparison ───────────────────────────────")
        cw = cross_wave_comparison(df, metric="word_count")
        print(cw.to_string(index=False))

    print("\n" + "=" * 70 + "\n")

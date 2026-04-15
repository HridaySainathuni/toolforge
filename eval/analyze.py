"""
Analyze ToolForge evaluation results and produce plots.

Usage:
    python -m eval.analyze --results eval/results/
    python -m eval.analyze --results eval/results/ --out eval/plots/

Produces:
    metrics_summary.csv  — per-condition/domain metrics table
    success_rate.png     — bar chart: success rate by domain × condition
    reuse_over_time.png  — line chart: cumulative reuse rate over tasks (full condition)
    cross_domain.png     — bar chart: success rate by domain (full condition only)
    ablation.png         — bar chart: mean success rate by condition
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all CSV files from results_dir into a single DataFrame."""
    dfs = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".csv") and not fname.startswith("metrics"):
            path = os.path.join(results_dir, fname)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    if not dfs:
        raise ValueError(f"No CSV result files found in {results_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    # Normalize boolean columns (CSV may store as string "True"/"False")
    for col in ("success", "tool_created", "tool_reused"):
        if col in combined.columns:
            combined[col] = combined[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
            )
    return combined


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(condition, domain) metrics."""
    rows = []
    for (condition, domain), g in df.groupby(["condition", "domain"]):
        n = len(g)
        n_success = int(g["success"].sum())
        n_created = int(g["tool_created"].sum())
        n_reused = int(g["tool_reused"].sum())
        denominator = max(n_created + n_reused, 1)
        rows.append(
            {
                "condition": condition,
                "domain": domain,
                "n_tasks": n,
                "task_success_rate": round(n_success / n, 4) if n else 0.0,
                "tool_creation_rate": round(n_created / n, 4) if n else 0.0,
                "tool_reuse_rate": round(n_reused / denominator, 4),
                "avg_time_seconds": round(g["time_seconds"].mean(), 2),
                "avg_attempts": round(g["attempts"].mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_success_rate(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: task success rate by domain and condition."""
    metrics = compute_metrics(df)
    pivot = metrics.pivot_table(
        index="domain", columns="condition", values="task_success_rate", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Task Success Rate by Domain and Condition")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate.png"), dpi=150)
    plt.close()


def plot_reuse_over_time(df: pd.DataFrame, out_dir: str) -> None:
    """Line chart: cumulative tool reuse rate as tasks accumulate (full condition)."""
    full = df[df["condition"] == "full"].copy()
    if full.empty:
        print("Warning: no 'full' condition data — skipping reuse_over_time plot")
        return

    full = full.reset_index(drop=True)
    cumulative_reused = full["tool_reused"].cumsum()
    cumulative_tool_events = (full["tool_reused"] | full["tool_created"]).cumsum()
    cumulative_tool_events = cumulative_tool_events.clip(lower=1)
    cumulative_reuse_rate = cumulative_reused / cumulative_tool_events

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(full)), cumulative_reuse_rate)
    ax.set_title("Cumulative Tool Reuse Rate Over Time (Full Condition)\n(H4: does reuse increase as library matures?)")
    ax.set_xlabel("Task Number")
    ax.set_ylabel("Cumulative Reuse Rate")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reuse_over_time.png"), dpi=150)
    plt.close()


def plot_cross_domain(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: success rate by domain for full condition (H1: cross-domain transfer)."""
    full = df[df["condition"] == "full"]
    if full.empty:
        print("Warning: no 'full' condition data — skipping cross_domain plot")
        return

    metrics = compute_metrics(full)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics, x="domain", y="task_success_rate", ax=ax)
    ax.set_title("Cross-Domain Transfer: Success Rate by Domain (Full Condition)\n(H1: do math tools generalize to mixed tasks?)")
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_domain.png"), dpi=150)
    plt.close()


def plot_ablation(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart: overall success rate by condition (H2 + H3: what levers matter)."""
    metrics = compute_metrics(df)
    agg = metrics.groupby("condition")["task_success_rate"].mean().reset_index()
    agg = agg.rename(columns={"task_success_rate": "mean_success_rate"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=agg, x="condition", y="mean_success_rate", ax=ax)
    ax.set_title("Ablation Study: Mean Task Success Rate by Condition\n(H2: does abstraction prompt matter? H3: does librarian matter?)")
    ax.set_ylabel("Mean Success Rate")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation.png"), dpi=150)
    plt.close()


def analyze(results_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_all_results(results_dir)
    metrics = compute_metrics(df)

    print(f"\n=== Loaded {len(df)} task results across {df['condition'].nunique()} condition(s) ===\n")
    print(metrics.to_string(index=False))

    metrics_path = os.path.join(out_dir, "metrics_summary.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    plot_success_rate(df, out_dir)
    plot_reuse_over_time(df, out_dir)
    plot_cross_domain(df, out_dir)
    plot_ablation(df, out_dir)
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ToolForge evaluation results")
    parser.add_argument("--results", default="eval/results/", help="Directory with CSV result files")
    parser.add_argument("--out", default="eval/plots/", help="Output directory for plots and metrics")
    args = parser.parse_args()
    analyze(args.results, args.out)

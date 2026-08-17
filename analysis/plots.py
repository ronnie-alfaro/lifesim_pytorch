from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_run_plots(agents: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    for agent_id, group in agents.groupby("agent_id"):
        ax.plot(group["tick"], group["cumulative_reward"], label=agent_id, linewidth=1)
    ax.set(title="Cumulative reward per agent", xlabel="Tick", ylabel="Reward")
    ax.legend(fontsize=6, ncol=3)
    generated.append(_save(fig, output_dir / "reward.png"))

    losses = agents.dropna(subset=["loss"])
    fig, ax = plt.subplots(figsize=(10, 5))
    if not losses.empty:
        for agent_id, group in losses.groupby("agent_id"):
            ax.plot(group["tick"], group["loss"], alpha=0.5, linewidth=0.8, label=agent_id)
    ax.set(title="Loss over training", xlabel="Tick", ylabel="Smooth L1 loss")
    generated.append(_save(fig, output_dir / "loss.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary["tick"], summary["living_humans"], label="Humans")
    ax.plot(summary["tick"], summary["living_animals"], label="Animals")
    ax.set(title="Agents alive over time", xlabel="Tick", ylabel="Alive")
    ax.legend()
    generated.append(_save(fig, output_dir / "survival.png"))

    counts = agents[agents["action"] != "DEAD"].groupby(["agent_type", "action"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    if not counts.empty:
        counts.T.plot(kind="bar", ax=ax)
    ax.set(title="Action distribution", xlabel="Action", ylabel="Count")
    generated.append(_save(fig, output_dir / "actions.png"))

    survived = agents.groupby(["agent_type", "agent_id"])["alive"].sum().groupby("agent_type").mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    survived.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set(title="Average ticks survived", xlabel="Agent type", ylabel="Ticks")
    generated.append(_save(fig, output_dir / "average_survival.png"))

    progression = agents.groupby(["tick", "agent_type"])["reward"].mean().unstack()
    window = max(2, min(100, len(summary) // 10))
    fig, ax = plt.subplots(figsize=(10, 5))
    progression.rolling(window, min_periods=1).mean().plot(ax=ax)
    ax.set(title=f"Reward progression ({window}-tick moving average)", xlabel="Tick", ylabel="Mean reward")
    generated.append(_save(fig, output_dir / "reward_progression.png"))
    return generated


def generate_comparison_plots(results_experiment_dir: Path) -> list[Path]:
    rows: list[dict[str, float | int]] = []
    for run_dir in sorted(results_experiment_dir.glob("run_*")):
        summary_path = run_dir / "summary.csv"
        agent_path = run_dir / "agents.csv"
        metadata_path = run_dir / "run_summary.json"
        if not (summary_path.is_file() and agent_path.is_file() and metadata_path.is_file()):
            continue
        run_number = int(run_dir.name.split("_")[-1])
        summary_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append({
            "run": run_number,
            "mean_reward": float(summary_data["mean_total_reward"]),
            "mean_survival": float(summary_data["mean_survival"]),
        })
    if len(rows) < 2:
        return []
    frame = pd.DataFrame(rows).sort_values("run")
    output_dir = results_experiment_dir / "comparisons"
    output_dir.mkdir(exist_ok=True)
    generated: list[Path] = []
    for column, title, filename in [
        ("mean_reward", "Average reward by run", "reward_by_run.png"),
        ("mean_survival", "Average survival by run", "survival_by_run.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frame["run"], frame[column], marker="o")
        ax.set(title=title, xlabel="Run", ylabel=column.replace("_", " "))
        ax.set_xticks(frame["run"])
        generated.append(_save(fig, output_dir / filename))
    return generated


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

"""Best Result Brain (BRB) registry and immutable champion packages."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from config import SimulationConfig
from persistence.checkpoints import read_metadata


BRB_VERSION = 1
BRB_SCORE_VERSION = 2


def ensure_best_result_brain(root: Path) -> dict[str, Any] | None:
    """Return the BRB registry, bootstrapping it from completed historical runs."""
    registry_path = root / "checkpoints" / "best_result_brain" / "registry.json"
    if registry_path.is_file():
        registry = _read_registry(registry_path, allow_legacy=True)
        if (
            registry.get("version") == BRB_VERSION
            and registry.get("score_version") == BRB_SCORE_VERSION
        ):
            return registry
    candidates: list[tuple[tuple[float, ...], Path, dict[str, Any]]] = []
    for summary_path in (root / "results").glob("experiment_*/run_*/run_summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("termination_reason") == "user_interrupt":
                continue
            checkpoint_dir = (
                root / "checkpoints" / summary_path.parent.parent.name / summary_path.parent.name
            )
            if (
                (checkpoint_dir / "metadata.json").is_file()
                and _uses_current_learning_contract(checkpoint_dir)
            ):
                candidates.append((_score(summary), checkpoint_dir, summary))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    _, checkpoint_dir, summary = max(candidates, key=lambda item: item[0])
    return _promote(root, checkpoint_dir, summary)


def consider_run_for_brb(
    root: Path, checkpoint_dir: Path, summary: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """Promote a completed run only when its group score beats the champion."""
    if summary.get("termination_reason") == "user_interrupt":
        return ensure_best_result_brain(root), False
    registry_path = root / "checkpoints" / "best_result_brain" / "registry.json"
    registry_existed = registry_path.is_file()
    current = ensure_best_result_brain(root)
    if (
        not registry_existed
        and current is not None
        and Path(current["source_checkpoint"]).resolve() == checkpoint_dir.resolve()
    ):
        return current, True
    candidate_score = _score(summary)
    if current is not None and candidate_score <= tuple(current["score_vector"]):
        return current, False
    return _promote(root, checkpoint_dir, summary), True


def brb_public_summary(root: Path) -> dict[str, Any] | None:
    registry = ensure_best_result_brain(root)
    if registry is None:
        return None
    return {
        "experiment_id": registry["experiment_id"],
        "run_number": registry["run_number"],
        "source_checkpoint": registry["source_checkpoint"],
        "mean_human_survival": registry["score"]["mean_human_survival"],
        "median_human_survival": registry["score"]["median_human_survival"],
        "last_human_tick": registry["score"]["last_human_tick"],
        "final_humans": registry["score"]["final_humans"],
        "initial_humans": registry["score"]["initial_humans"],
        "human_completion_ratio": registry["score"]["human_completion_ratio"],
        "brain_selected_drinks": registry["score"]["brain_selected_drinks"],
        "human_architecture": registry["architectures"]["human"],
        "animal_architecture": registry["architectures"]["animal"],
        "minimum_human_width": _minimum_slider_width(
            registry["architectures"]["human"]
        ),
        "minimum_animal_width": _minimum_slider_width(
            registry["architectures"]["animal"]
        ),
        "learning_contract_current": _same_learning_contract(
            registry.get("config", {}), SimulationConfig().to_dict()
        ),
        "source_reward_version": int(
            registry.get("config", {}).get("reward_version", 1)
        ),
        "current_reward_version": SimulationConfig().reward_version,
    }


def load_brb_payloads(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    registry = ensure_best_result_brain(root)
    if registry is None:
        raise RuntimeError("No completed run is available as Best Result Brain")
    champion_dir = root / registry["champion_dir"]
    payloads: dict[str, list[dict[str, Any]]] = {"human": [], "animal": []}
    for row in registry["agents"]:
        payload = torch.load(
            champion_dir / row["file"], map_location="cpu", weights_only=False
        )
        payloads[row["agent_type"]].append(payload)
    for values in payloads.values():
        values.sort(
            key=lambda value: (
                int(value["steps_survived"]), float(value["total_reward"])
            ),
            reverse=True,
        )
    return registry, payloads


def _promote(
    root: Path, checkpoint_dir: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    metadata = read_metadata(checkpoint_dir)
    experiment_id = int(metadata["experiment_id"])
    run_number = int(metadata["run_number"])
    relative_dir = Path("checkpoints") / "best_result_brain" / "champions" / (
        f"experiment_{experiment_id:03d}_run_{run_number:03d}"
    )
    champion_dir = root / relative_dir
    champion_dir.mkdir(parents=True, exist_ok=True)
    agent_rows: list[dict[str, Any]] = []
    architectures: dict[str, dict[str, Any]] = {}
    for filename in metadata["agent_files"]:
        source = torch.load(
            checkpoint_dir / filename, map_location="cpu", weights_only=False
        )
        slim = {
            "agent_id": source["agent_id"],
            "agent_type": source["agent_type"],
            "architecture": source["architecture"],
            "model_state_dict": source["model_state_dict"],
            "steps_survived": int(source.get("steps_survived", 0)),
            "total_reward": float(source.get("total_reward", 0.0)),
        }
        target_name = f"{source['agent_id']}.pt"
        torch.save(slim, champion_dir / target_name)
        architectures.setdefault(source["agent_type"], source["architecture"])
        agent_rows.append({
            "agent_id": source["agent_id"],
            "agent_type": source["agent_type"],
            "file": target_name,
            "steps_survived": slim["steps_survived"],
            "total_reward": slim["total_reward"],
        })
    score_vector = list(_score(summary))
    # Historical summaries lacked individual values. Recover their true
    # tick-level median from metrics instead of using the death tick counter.
    if "individual_survival" not in summary["by_type"]["human"]:
        historical_survival = _historical_human_survival(root, metadata)
        if historical_survival:
            score_vector[3] = float(statistics.median(historical_survival))
    registry = {
        "version": BRB_VERSION,
        "score_version": BRB_SCORE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "run_number": run_number,
        "source_checkpoint": str(checkpoint_dir.resolve()),
        "champion_dir": str(relative_dir),
        "score_vector": score_vector,
        "score": {
            "completed_tick_limit": bool(score_vector[0]),
            "human_completion_ratio": score_vector[1],
            "final_humans": int(summary.get("final_humans", 0)),
            "initial_humans": int(summary.get("initial_humans", 0)),
            "mean_human_survival": score_vector[4],
            "median_human_survival": score_vector[3],
            "brain_selected_drinks": int(score_vector[5]),
            "successful_meals": int(score_vector[6]),
            "last_human_tick": int(summary["ticks_executed"]),
        },
        "config": metadata["config"],
        "architectures": architectures,
        "agents": agent_rows,
    }
    registry_path = root / "checkpoints" / "best_result_brain" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


def _score(summary: dict[str, Any]) -> tuple[float, ...]:
    human = summary["by_type"]["human"]
    survival_values = []
    # Older summaries do not contain individual values; mean remains primary.
    if "individual_survival" in human:
        survival_values = list(human["individual_survival"].values())
    median_survival = (
        float(statistics.median(survival_values))
        if survival_values
        else float(human["mean_survival"])
    )
    initial_humans = max(1, int(summary.get("initial_humans", 1)))
    final_humans = max(0, int(summary.get("final_humans", 0)))
    completed_tick_limit = summary.get("termination_reason") == "tick_limit"
    completion_ratio = final_humans / initial_humans if completed_tick_limit else 0.0
    # A run that reaches its full horizon with humans alive is a successful
    # survival experiment. Rank that first, then group robustness, then the
    # older learning indicators. This prevents a long extinction run from
    # displacing a genuinely successful BRB.
    return (
        float(completed_tick_limit and final_humans > 0),
        float(completion_ratio),
        float(final_humans),
        median_survival,
        float(human["mean_survival"]),
        float(human["brain_selected_drinks"]),
        float(human["successful_meals"]),
        -float(human.get("ignored_survival_priority_actions", 0)),
    )


def _minimum_slider_width(architecture: dict[str, Any]) -> int:
    need, spatial, fusion = [int(value) for value in architecture["hidden_sizes"]]
    return max(spatial, fusion, need * 2)


def _historical_human_survival(
    root: Path, metadata: dict[str, Any]
) -> list[int]:
    import csv

    path = (
        root
        / "results"
        / f"experiment_{int(metadata['experiment_id']):03d}"
        / f"run_{int(metadata['run_number']):03d}"
        / "agents.csv"
    )
    if not path.is_file():
        return []
    totals: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("agent_type") != "human":
                continue
            alive = str(row.get("alive", "")).lower() in {"true", "1"}
            totals[row["agent_id"]] = totals.get(row["agent_id"], 0) + int(alive)
    return list(totals.values())


def _uses_current_learning_contract(checkpoint_dir: Path) -> bool:
    """Do not compare scores produced by older reward/brain contracts."""
    metadata = read_metadata(checkpoint_dir)
    config = metadata.get("config", {})
    current = SimulationConfig()
    return _same_learning_contract(config, current.to_dict())


def _same_learning_contract(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        int(first.get("reward_version", 1))
        == int(second.get("reward_version", 1))
        and int(first.get("brain_architecture_version", 1))
        == int(second.get("brain_architecture_version", 1))
    )


def _read_registry(path: Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read BRB registry: {error}") from error
    if payload.get("version") != BRB_VERSION and not allow_legacy:
        raise RuntimeError("Unsupported BRB registry version")
    return payload

from __future__ import annotations

from pathlib import Path

from config import SimulationConfig
from persistence.best_result import load_brb_payloads
from persistence.checkpoints import (
    _load_widened_state_dict,
    load_agents,
    read_metadata,
)
from simulation.engine import RunResult, SimulationEngine, set_seed
from world.world import World


def next_experiment_id(root: Path) -> int:
    ids = []
    for path in (root / "checkpoints").glob("experiment_*"):
        try:
            ids.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    return max(ids, default=0) + 1


def start_new(root: Path, config: SimulationConfig, seed: int) -> RunResult:
    return build_new_engine(root, config, seed).run()


def build_new_engine(root: Path, config: SimulationConfig, seed: int) -> SimulationEngine:
    set_seed(seed)
    experiment_id = next_experiment_id(root)
    world = World(config)
    return SimulationEngine(world, config, experiment_id, 1, seed, root)


def build_brb_engine(root: Path, config: SimulationConfig, seed: int) -> SimulationEngine:
    """Start an independent experiment from the current BRB champion weights.

    Physical state, Adam, personal replay and Horde replay intentionally start
    empty. If the requested population is larger, the strongest saved brains
    are reused cyclically as parents for the new independent agents.
    """
    set_seed(seed)
    registry, payloads = load_brb_payloads(root)
    world = World(config)
    parents: dict[str, str] = {}
    for agent in world.agents:
        candidates = payloads[agent.agent_type]
        if not candidates:
            raise RuntimeError(f"BRB has no {agent.agent_type} brains")
        same_type_index = int(agent.id.rsplit("_", 1)[-1]) - 1
        source = candidates[same_type_index % len(candidates)]
        _validate_brb_architecture(agent.brain.architecture, source["architecture"])
        _load_widened_state_dict(agent.brain, source["model_state_dict"])
        agent.trainer.target_brain.load_state_dict(agent.brain.state_dict(), strict=True)
        parents[agent.id] = str(source["agent_id"])
    return SimulationEngine(
        world,
        config,
        next_experiment_id(root),
        1,
        seed,
        root,
        brb_source=str(registry["source_checkpoint"]),
        brb_parents=parents,
    )


def _validate_brb_architecture(
    target: dict[str, object], source: dict[str, object]
) -> None:
    source_hidden = [int(value) for value in source["hidden_sizes"]]
    target_hidden = [int(value) for value in target["hidden_sizes"]]
    compatible = (
        int(target["architecture_version"]) == int(source["architecture_version"])
        and int(target["input_size"]) >= int(source["input_size"])
        and int(target["need_input_size"]) >= int(source["need_input_size"])
        and int(target["output_size"]) >= int(source["output_size"])
        and len(target_hidden) == len(source_hidden)
        and all(new >= old for new, old in zip(target_hidden, source_hidden))
    )
    if not compatible:
        required = max(source_hidden[1], source_hidden[2], source_hidden[0] * 2)
        raise ValueError(
            "The selected brain is too small or incompatible with BRB; "
            f"use width {required} or greater, or disable BRB"
        )


def resume(root: Path, checkpoint_dir: Path, config: SimulationConfig, seed: int) -> RunResult:
    return build_resumed_engine(root, checkpoint_dir, config, seed).run()


def run_continuous(
    root: Path,
    config: SimulationConfig,
    seed: int,
    checkpoint_dir: Path | None = None,
    max_runs: int | None = None,
) -> list[RunResult]:
    """Chain learned brains into new runs until Ctrl+C (or a test limit)."""
    engine = (
        build_resumed_engine(root, checkpoint_dir, config, seed)
        if checkpoint_dir is not None
        else build_new_engine(root, config, seed)
    )
    completed: list[RunResult] = []
    previous_tick = _previous_termination_tick(checkpoint_dir)
    print("\nLIFESIM CONTINUO · solo texto · Ctrl+C para detener")
    while max_runs is None or len(completed) < max_runs:
        previous = "ninguno" if previous_tick is None else f"tick {previous_tick}"
        print(
            f"\nCYCLE START | experimento {engine.experiment_id:03d} | "
            f"run {engine.run_number:03d} | seed {engine.seed} | "
            f"run anterior terminó en: {previous}"
        )
        if engine.learning_state_resets:
            print(
                "REWARD MIGRATION | se conservaron los pesos, pero se limpiaron "
                "replay/Adam/target; cada agente conserva su perfil epsilon Horde"
            )
        try:
            result = engine.run()
        except KeyboardInterrupt:
            if engine.current_tick > 0 and not engine.is_finalized:
                if not engine.is_complete:
                    engine.termination_reason = "user_interrupt"
                result = engine.finalize()
                completed.append(result)
                _print_continuous_run_end(engine)
            print("\nSIMULACIÓN DETENIDA POR EL USUARIO")
            if completed:
                print(f"Último checkpoint seguro: {completed[-1].checkpoint_dir}")
            return completed

        completed.append(result)
        _print_continuous_run_end(engine)
        previous_tick = engine.current_tick
        if max_runs is not None and len(completed) >= max_runs:
            break
        seed += 1
        engine = build_resumed_engine(root, result.checkpoint_dir, config, seed)
    return completed


def build_resumed_engine(
    root: Path, checkpoint_dir: Path, config: SimulationConfig, seed: int
) -> SimulationEngine:
    checkpoint_dir = checkpoint_dir.resolve()
    metadata = read_metadata(checkpoint_dir)
    experiment_id = int(metadata["experiment_id"])
    existing_runs = [
        int(path.name.split("_")[-1])
        for path in checkpoint_dir.parent.glob("run_*")
        if path.name.split("_")[-1].isdigit()
    ]
    run_number = max(existing_runs, default=int(metadata["run_number"])) + 1
    set_seed(seed)
    world = World(config)
    agent_files = metadata.get("agent_files", [])
    positions = world.random_empty_positions(len(agent_files))
    loaded_agents, loaded_hashes, migrated_agents = load_agents(
        checkpoint_dir, config, positions
    )
    learning_state_resets = [
        agent.id for agent in loaded_agents
        if getattr(agent, "learning_state_reset", False)
    ]
    world.agents = loaded_agents
    world.initialize_stockpiles()
    expected_hashes = metadata.get("final_model_hashes", {})
    exact_ids = set(loaded_hashes) - set(migrated_agents)
    if any(loaded_hashes[agent_id] != expected_hashes.get(agent_id) for agent_id in exact_ids):
        raise RuntimeError("Run initial weights do not exactly match source final weights")
    if not config.compact_console:
        print(f"Loaded {len(loaded_agents)} independent brains from {checkpoint_dir}")
    if migrated_agents and not config.compact_console:
        print(
            f"Widened {len(migrated_agents)} brains while preserving their initial outputs; "
            "their optimizers were safely reinitialized."
        )
    if learning_state_resets and not config.compact_console:
        print(
            f"Reward changed: reset replay, Adam, target network and exploration for "
            f"{len(learning_state_resets)} agents while preserving their brain weights."
        )
    return SimulationEngine(
        world, config, experiment_id, run_number, seed, root, checkpoint_dir,
        architecture_migrations=migrated_agents,
        learning_state_resets=learning_state_resets,
    )


def _previous_termination_tick(checkpoint_dir: Path | None) -> int | None:
    if checkpoint_dir is None:
        return None
    metadata = read_metadata(checkpoint_dir.resolve())
    value = metadata.get("ticks_executed")
    return int(value) if value is not None else None


def _print_continuous_run_end(engine: SimulationEngine) -> None:
    summary = engine.learning_summary
    if not isinstance(summary, dict):
        return
    humans = summary["by_type"]["human"]
    reason = {
        "human_extinction": "murió el último humano",
        "tick_limit": "alcanzó el límite",
        "user_interrupt": "interrumpido por el usuario",
        "user_cancelled": "cancelado por el usuario",
    }.get(str(summary["termination_reason"]), str(summary["termination_reason"]))
    print(
        f"CYCLE END   | run {engine.run_number:03d} | tick {summary['ticks_executed']} | "
        f"{reason} | humanos {summary['final_humans']}/{summary['initial_humans']} | "
        f"bebidas {humans['successful_drinks']} "
        f"({humans['brain_selected_drinks']} brain) | "
        f"muertes con sed crítica {humans['deaths_with_critical_thirst']} | "
        f"supervivencia media {humans['mean_survival']:.1f}"
    )
    comparison = summary.get("comparison_to_previous_run")
    if isinstance(comparison, dict):
        print(
            "COMPARACIÓN | último humano "
            f"{comparison['previous_last_human_tick']} -> "
            f"{comparison['current_last_human_tick']} "
            f"({comparison['last_human_tick_change']:+d}) | media "
            f"{comparison['previous_mean_human_survival']:.1f} -> "
            f"{comparison['current_mean_human_survival']:.1f}"
        )

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from agents.animal import Animal
from agents.human import Human
from config import SimulationConfig
from learning.replay_buffer import ReplayBuffer
from learning.rewards import calculate_reward
from persistence.checkpoints import (
    load_horde_replay_state,
    model_hash,
    save_run_checkpoints,
)
from simulation.metrics import MetricsRecorder
from world.grid import Action
from world.renderer import AsciiRenderer
from world.world import World


@dataclass(frozen=True)
class RunResult:
    checkpoint_dir: Path
    results_dir: Path
    plot_paths: list[Path]
    initial_hashes: dict[str, str]
    final_hashes: dict[str, str]


class SimulationEngine:
    def __init__(
        self, world: World, config: SimulationConfig, experiment_id: int,
        run_number: int, seed: int, root: Path,
        source_checkpoint: Path | None = None,
        architecture_migrations: list[str] | None = None,
        learning_state_resets: list[str] | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.experiment_id = experiment_id
        self.run_number = run_number
        self.seed = seed
        self.root = root
        self.source_checkpoint = source_checkpoint
        self.architecture_migrations = architecture_migrations or []
        self.learning_state_resets = learning_state_resets or []
        self.metrics = MetricsRecorder()
        self.renderer = AsciiRenderer()
        self.current_tick = 0
        self.termination_reason: str | None = None
        self.initial_hashes = {agent.id: model_hash(agent) for agent in self.world.agents}
        self.horde_replay_buffers = self._initialize_horde_learning()
        self.last_agent_events: dict[str, dict[str, object]] = {}
        self.learning_summary: dict[str, object] | None = None
        self._result: RunResult | None = None

    def run(self) -> RunResult:
        while self.step():
            pass
        return self.finalize()

    def step(self) -> bool:
        """Execute one complete learning tick.

        Returns ``False`` when no more ticks can be executed. The web server
        calls this exact method slowly, so visualization and training never
        diverge into separate simulations.
        """
        if self._result is not None or self.is_complete:
            return False
        tick = self.current_tick + 1
        self.world.update_resources()
        tick_rewards: dict[str, float] = {}
        tick_losses: dict[str, float | None] = {}
        tick_actions: dict[str, str] = {}
        tick_components: dict[str, dict[str, float]] = {}
        events: dict[str, dict[str, object]] = {}
        for agent in list(self.world.living_agents):
            # WORLD -> PERCEPTION -> BRAIN/ACTION
            state = agent.perceive(self.world)
            urgent_resource: str | None = None
            if agent.thirst >= 0.25 or agent.hunger >= 0.25:
                urgent_resource = "water" if agent.thirst >= agent.hunger else "food"
            distance_before = (
                self.world.distance_to_nearest(agent.x, agent.y, urgent_resource)
                if urgent_resource is not None
                else None
            )
            action_index = agent.choose_action(
                state, (self.run_number - 1) * self.config.num_ticks + tick
            )
            action_result = self.world.execute_action(agent, action_index)
            distance_after = (
                self.world.distance_to_nearest(agent.x, agent.y, urgent_resource)
                if urgent_resource is not None
                else None
            )
            resource_progress = 0.0
            movement_actions = {
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
            }
            if (
                Action(action_index) in movement_actions
                and distance_before is not None
                and distance_after is not None
            ):
                if distance_after < distance_before:
                    resource_progress = 0.05
                elif distance_after > distance_before:
                    resource_progress = -0.03
            agent.note_position()
            health_delta, died = agent.update_biology(rested=action_result.rested)

            unnecessary_penalty = agent.unnecessary_need_penalty(
                Action(action_index).name,
                action_result.unnecessary_need_action,
            )
            need_reward = action_result.necessity * self.config.need_reward_scale

            # ACTION -> REWARD (components remain observable in the web UI/debug mode)
            reward_result = calculate_reward(
                ate=action_result.ate,
                drank=action_result.drank,
                invalid=action_result.invalid,
                reached_needed_resource=action_result.reached_needed_resource,
                health_delta=health_delta,
                died=died,
                repeating=agent.is_repeating_position,
                resource_progress=resource_progress,
                rested=action_result.rested,
                need_reward=need_reward,
                unnecessary_action_penalty=unnecessary_penalty,
            )
            agent.total_reward += reward_result.total
            next_state = agent.perceive(self.world)

            # REWARD -> REPLAY BUFFER -> BACKWARD/optimizer.step
            loss = agent.remember_and_learn(
                state, action_index, reward_result.total, next_state, died
            )
            action_name = Action(action_index).name
            tick_rewards[agent.id] = reward_result.total
            tick_losses[agent.id] = loss
            tick_actions[agent.id] = action_name
            tick_components[agent.id] = reward_result.components
            events[agent.id] = {
                "agent_id": agent.id,
                "tick": tick,
                "action": action_name,
                "reward": reward_result.total,
                "reward_components": reward_result.components,
                "loss": loss,
                "trained": loss is not None,
                "training_steps": agent.trainer.training_steps,
                "replay_size": len(agent.trainer.replay_buffer),
                "horde_replay_size": len(agent.trainer.learning_replay_buffer),
                "epsilon": agent.last_epsilon,
                "exploration": agent.last_was_exploration,
                "observation": [float(value) for value in state.tolist()],
                "q_values": agent.last_q_values,
            }
            if self.config.debug_rewards:
                print(
                    f"DEBUG tick={tick} agent={agent.id} action={action_name} "
                    f"reward={reward_result.components}"
                )
        self.last_agent_events = events
        self.world.validate()
        self.metrics.record(
            tick,
            self.world,
            tick_rewards,
            tick_losses,
            tick_actions,
            tick_components,
            events,
        )
        self.current_tick = tick
        living_humans = sum(
            agent.alive for agent in self.world.agents if agent.agent_type == "human"
        )
        if self.config.stop_when_no_humans and living_humans == 0:
            self.termination_reason = "human_extinction"
        elif tick >= self.config.num_ticks:
            self.termination_reason = "tick_limit"
        if self.config.render_every and tick % self.config.render_every == 0:
            print(self.renderer.render(self.world))
        if tick == 1 or tick % self.config.status_every == 0 or self.is_complete:
            self._print_status(tick)
        return True

    def finalize(self) -> RunResult:
        """Persist metrics, plots, and brains once the last tick is complete."""
        if self._result is not None:
            return self._result
        if not self.is_complete:
            raise RuntimeError("Cannot finalize a run before a terminal condition")
        experiment_name = f"experiment_{self.experiment_id:03d}"
        run_name = f"run_{self.run_number:03d}"
        results_dir = self.root / "results" / experiment_name / run_name
        checkpoint_dir = self.root / "checkpoints" / experiment_name / run_name
        agents_path, summary_path = self.metrics.save(results_dir)
        plot_paths: list[Path] = []
        comparison_paths: list[Path] = []
        if self.config.generate_plots:
            # Plotting is optional in high-throughput text-only experiments.
            from analysis.plots import generate_comparison_plots, generate_run_plots

            plot_paths = generate_run_plots(
                self.metrics.agent_frame(), self.metrics.summary_frame(), results_dir
            )
        checkpoint_path = save_run_checkpoints(
            self.world.agents, checkpoint_dir, self.experiment_id, self.run_number,
            self.seed, self.config, str(self.source_checkpoint) if self.source_checkpoint else None,
            self.initial_hashes,
            termination_reason=self.termination_reason,
            ticks_executed=self.current_tick,
            horde_replay_buffers=self.horde_replay_buffers,
        )
        final_hashes = {agent.id: model_hash(agent) for agent in self.world.agents}
        summary_data = self._write_run_summary(results_dir)
        self.learning_summary = summary_data
        if self.config.generate_plots:
            comparison_paths = generate_comparison_plots(results_dir.parent)
        if not self.config.compact_console:
            self._print_learning_summary(summary_data)
            print(f"Metrics: {agents_path}, {summary_path}")
            print(f"Checkpoint: {checkpoint_path}")
            if plot_paths or comparison_paths:
                print(
                    "Plots generated: "
                    f"{', '.join(path.name for path in plot_paths + comparison_paths)}"
                )
        self._result = RunResult(
            checkpoint_path,
            results_dir,
            plot_paths + comparison_paths,
            self.initial_hashes,
            final_hashes,
        )
        return self._result

    @property
    def is_complete(self) -> bool:
        return self.termination_reason is not None

    @property
    def is_finalized(self) -> bool:
        return self._result is not None

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe, read-only view for renderers and web clients."""
        latest_summary = self.metrics.summary_rows[-1] if self.metrics.summary_rows else {
            "living_humans": self.config.num_humans,
            "living_animals": self.config.num_animals,
            "average_human_reward": 0.0,
            "average_animal_reward": 0.0,
            "average_loss": None,
            "food_remaining": len(self.world.food),
            "water_remaining": len(self.world.water),
            "deaths": 0,
        }
        agents = []
        for agent in self.world.agents:
            event = self.last_agent_events.get(agent.id, {})
            agent_class = Human if agent.agent_type == "human" else Animal
            agents.append({
                "id": agent.id,
                "type": agent.agent_type,
                "x": agent.x,
                "y": agent.y,
                "alive": agent.alive,
                "health": agent.health,
                "energy": agent.energy,
                "hunger": agent.hunger,
                "thirst": agent.thirst,
                "total_reward": agent.total_reward,
                "steps_survived": agent.steps_survived,
                "action": event.get("action", "WAITING" if agent.alive else "DEAD"),
                "reward": event.get("reward", 0.0),
                "loss": event.get("loss"),
                "trained": event.get("trained", False),
                "training_steps": agent.trainer.training_steps,
                "replay_size": len(agent.trainer.replay_buffer),
                "horde_replay_size": len(agent.trainer.learning_replay_buffer),
                "horde_learning": agent.trainer.horde_replay_buffer is not None,
                "epsilon": event.get("epsilon", agent.last_epsilon),
                "exploration_profile": agent.exploration_profile,
                "exploration": event.get("exploration", False),
                "observation": event.get("observation", []),
                "q_values": event.get("q_values", agent.last_q_values),
                "reward_components": event.get("reward_components", {}),
                "brain_architecture": agent.brain.architecture,
                "brain_activations": agent.last_brain_activations,
                "weight_statistics": agent.brain.weight_statistics(),
                "need_labels": list(agent_class.NEED_LABELS),
                "spatial_labels": list(agent_class.SPATIAL_LABELS),
                "spatial_memory": agent.spatial_memory.snapshot(agent),
            })
        return {
            "experiment_id": self.experiment_id,
            "run_number": self.run_number,
            "seed": self.seed,
            "tick": self.current_tick,
            "num_ticks": self.config.num_ticks,
            "complete": self.is_complete,
            "termination_reason": self.termination_reason,
            "learning_summary": self.learning_summary,
            "source_checkpoint": str(self.source_checkpoint) if self.source_checkpoint else None,
            "architecture_migrations": self.architecture_migrations,
            "learning_state_resets": self.learning_state_resets,
            "experiment_config": {
                "brain_architecture_version": self.config.brain_architecture_version,
                "num_humans": self.config.num_humans,
                "num_animals": self.config.num_animals,
                "human_hidden_sizes": list(self.config.human_brain.hidden_sizes),
                "animal_hidden_sizes": list(self.config.animal_brain.hidden_sizes),
                "horde_learning_enabled": self.config.horde_learning_enabled,
                "epsilon_standard_range": [
                    self.config.epsilon_standard_min,
                    self.config.epsilon_standard_max,
                ],
                "epsilon_scout": self.config.epsilon_scout,
            },
            "horde": {
                "enabled": self.config.horde_learning_enabled,
                "human_replay_size": len(self.horde_replay_buffers.get("human", [])),
                "animal_replay_size": len(self.horde_replay_buffers.get("animal", [])),
            },
            "grid": {
                "width": self.world.width,
                "height": self.world.height,
                "food": [list(position) for position in sorted(self.world.food)],
                "water": [list(position) for position in sorted(self.world.water)],
                "obstacles": [list(position) for position in sorted(self.world.obstacles)],
            },
            "agents": agents,
            "summary": latest_summary,
        }

    def _initialize_horde_learning(self) -> dict[str, ReplayBuffer]:
        """Attach one persistent collective replay to every species."""
        if not self.config.horde_learning_enabled:
            return {}
        persisted = (
            load_horde_replay_state(self.source_checkpoint)
            if (
                self.source_checkpoint is not None
                and (self.source_checkpoint / "metadata.json").is_file()
            )
            else None
        )
        buffers: dict[str, ReplayBuffer] = {}
        for agent_type, input_size in (
            ("human", Human.INPUT_SIZE),
            ("animal", Animal.INPUT_SIZE),
        ):
            replay = ReplayBuffer(self.config.horde_replay_capacity)
            if persisted is not None:
                replay.load_state_dict(persisted[agent_type], input_size)
            elif self.source_checkpoint is not None:
                # First Horde run from an older Brain v2 checkpoint: merge all
                # personal histories so prior experience is not discarded.
                combined: list[dict[str, object]] = []
                for agent in self.world.agents:
                    if agent.agent_type == agent_type:
                        combined.extend(agent.trainer.replay_buffer.state_dict())
                replay.load_state_dict(combined, input_size)
            for agent in self.world.agents:
                if agent.agent_type == agent_type:
                    agent.trainer.join_horde(replay)
            buffers[agent_type] = replay
        return buffers

    def _print_status(self, tick: int) -> None:
        latest = self.metrics.summary_rows[-1]
        if self.config.compact_console:
            human_rows = self.metrics.agent_frame()
            human_rows = human_rows[human_rows["agent_type"] == "human"]
            drinks = int(human_rows["drank"].sum())
            epsilon = _mean_agent_epsilon(self.world.agents)
            print(
                f"Run {self.run_number:03d} | tick {tick:>5}/{self.config.num_ticks} | "
                f"H {latest['living_humans']}/{self.config.num_humans} | "
                f"A {latest['living_animals']}/{self.config.num_animals} | "
                f"bebidas H {drinks} | epsilon {epsilon:.3f} | "
                f"loss {_format_optional(latest['average_loss'])}"
            )
            return
        print(
            f"Experiment: {self.experiment_id:03d} | Run: {self.run_number:03d} | "
            f"Tick: {tick}/{self.config.num_ticks} | "
            f"Humans: {latest['living_humans']}/{self.config.num_humans} | "
            f"Animals: {latest['living_animals']}/{self.config.num_animals} | "
            f"Avg rewards H/A: {latest['average_human_reward']:.2f}/"
            f"{latest['average_animal_reward']:.2f} | Loss: "
            f"{_format_optional(latest['average_loss'])} | Food/Water: "
            f"{latest['food_remaining']}/{latest['water_remaining']}"
        )

    def _write_run_summary(self, results_dir: Path) -> dict[str, object]:
        frame = self.metrics.agent_frame()
        # Cumulative reward can rise and fall, so the run total is the final row,
        # not the maximum value observed along the way.
        total_by_agent = frame.groupby(["agent_type", "agent_id"])["cumulative_reward"].last()
        survival = frame.groupby(["agent_type", "agent_id"])["alive"].sum()
        payload: dict[str, object] = {
            "experiment_id": self.experiment_id,
            "run_number": self.run_number,
            "seed": self.seed,
            "mean_total_reward": float(total_by_agent.mean()),
            "mean_survival": float(survival.mean()),
            "by_type": {},
        }
        actual_ticks = max(1, self.current_tick)
        fifth = max(1, actual_ticks // 5)
        final_humans = int(
            frame[frame["agent_type"] == "human"].groupby("agent_id")["alive"].last().sum()
        )
        final_animals = int(
            frame[frame["agent_type"] == "animal"].groupby("agent_id")["alive"].last().sum()
        )
        payload.update({
            "ticks_executed": self.current_tick,
            "configured_ticks": self.config.num_ticks,
            "termination_reason": self.termination_reason,
            "initial_humans": self.config.num_humans,
            "final_humans": final_humans,
            "initial_animals": self.config.num_animals,
            "final_animals": final_animals,
        })
        by_type: dict[str, dict[str, float | str]] = {}
        for agent_type in ("human", "animal"):
            subset = frame[frame["agent_type"] == agent_type]
            reward_tick = subset.groupby("tick")["reward"].mean()
            loss_tick = subset.groupby("tick")["loss"].mean().dropna()
            first_reward = float(reward_tick.iloc[:fifth].mean())
            last_reward = float(reward_tick.iloc[-fifth:].mean())
            action_counts = subset[subset["action"] != "DEAD"]["action"].value_counts()
            most_common_action = str(action_counts.index[0]) if not action_counts.empty else "none"
            exploration_mask = subset["exploration"].fillna(False).astype(bool)
            by_type[agent_type] = {
                "first_20_percent_reward": first_reward,
                "last_20_percent_reward": last_reward,
                "reward_percent_change": _percent_change(first_reward, last_reward),
                "first_20_percent_loss": float(loss_tick.iloc[:max(1, len(loss_tick)//5)].mean()) if not loss_tick.empty else 0.0,
                "last_20_percent_loss": float(loss_tick.iloc[-max(1, len(loss_tick)//5):].mean()) if not loss_tick.empty else 0.0,
                "mean_survival": float(survival.loc[agent_type].mean()),
                "most_common_action": most_common_action,
                "successful_drinks": int(subset["drank"].sum()),
                "brain_selected_drinks": int(
                    (subset["drank"] & ~exploration_mask).sum()
                ),
                "exploration_drinks": int(
                    (subset["drank"] & exploration_mask).sum()
                ),
                "successful_meals": int(subset["ate"].sum()),
                "deaths_with_critical_thirst": int(
                    ((~subset.groupby("agent_id")["alive"].last().astype(bool))
                     & (subset.groupby("agent_id")["thirst"].last() >= 1.0)).sum()
                ),
                "deaths_with_critical_hunger": int(
                    ((~subset.groupby("agent_id")["alive"].last().astype(bool))
                     & (subset.groupby("agent_id")["hunger"].last() >= 1.0)).sum()
                ),
            }
        payload["by_type"] = by_type
        comparison = self._load_previous_run_comparison(by_type)
        payload["comparison_to_previous_run"] = comparison
        payload["what_they_learned"] = self._describe_learning(by_type, comparison)
        (results_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _print_learning_summary(self, payload: dict[str, object]) -> None:
        reason = {
            "human_extinction": "murió el último humano",
            "tick_limit": "se alcanzó el límite de ticks",
            "user_interrupt": "el usuario detuvo la simulación",
        }.get(str(payload["termination_reason"]), str(payload["termination_reason"]))
        print("\nLEARNING SUMMARY (evidencia descriptiva, no prueba definitiva)")
        print(
            f"Run detenido en tick {payload['ticks_executed']}/{payload['configured_ticks']}: {reason}."
        )
        print(
            f"Humanos: {payload['initial_humans']} -> {payload['final_humans']} | "
            f"Animales: {payload['initial_animals']} -> {payload['final_animals']}"
        )
        comparison = payload.get("comparison_to_previous_run")
        if isinstance(comparison, dict):
            print(
                "Comparación con run anterior: último humano "
                f"{comparison['previous_last_human_tick']} -> "
                f"{comparison['current_last_human_tick']} ticks "
                f"({comparison['last_human_tick_change']:+d}); supervivencia humana media "
                f"{comparison['previous_mean_human_survival']:.1f} -> "
                f"{comparison['current_mean_human_survival']:.1f}."
            )
        by_type = payload["by_type"]
        assert isinstance(by_type, dict)
        for name in ("human", "animal"):
            values = by_type[name]
            print(
                f"{name.title()}: reward first/last 20% "
                f"{values['first_20_percent_reward']:.4f} -> {values['last_20_percent_reward']:.4f} "
                f"({values['reward_percent_change']:+.1f}%), loss "
                f"{values['first_20_percent_loss']:.4f} -> {values['last_20_percent_loss']:.4f}, "
                f"mean survival {values['mean_survival']:.1f} ticks"
                f", drinks {values['successful_drinks']}, meals {values['successful_meals']}"
            )
        print("\n¿QUÉ APRENDIERON?")
        for sentence in payload["what_they_learned"]:
            print(f"- {sentence}")

    def _describe_learning(
        self,
        by_type: dict[str, dict[str, float | str]],
        comparison: dict[str, float | int] | None,
    ) -> list[str]:
        descriptions: list[str] = []
        for agent_type, label in (("human", "Los humanos"), ("animal", "Los animales")):
            values = by_type[agent_type]
            first_reward = float(values["first_20_percent_reward"])
            last_reward = float(values["last_20_percent_reward"])
            first_loss = float(values["first_20_percent_loss"])
            last_loss = float(values["last_20_percent_loss"])
            reward_delta = last_reward - first_reward
            if reward_delta > 0.005:
                reward_text = "recibieron mejores rewards"
            elif reward_delta < -0.005:
                reward_text = "recibieron peores rewards"
            else:
                reward_text = "mantuvieron rewards parecidos"
            if first_loss == 0.0 and last_loss == 0.0:
                loss_text = "y no reunieron experiencias suficientes para medir la loss"
            elif last_loss < first_loss * 0.95:
                loss_text = "y el error del brain bajó"
            elif last_loss > first_loss * 1.05:
                loss_text = "pero el error del brain subió"
            else:
                loss_text = "y el error del brain quedó casi igual"
            descriptions.append(
                f"{label} {reward_text} al final ({first_reward:.3f} -> {last_reward:.3f}) "
                f"{loss_text} ({first_loss:.4f} -> {last_loss:.4f})."
            )
            descriptions.append(
                f"Su acción más frecuente fue {_friendly_action(str(values['most_common_action']))}; "
                "esto describe su conducta, "
                "pero no demuestra por sí solo una estrategia útil."
            )
            successful_drinks = int(values["successful_drinks"])
            if successful_drinks:
                drink_word = "vez" if successful_drinks == 1 else "veces"
                brain_drinks = int(values["brain_selected_drinks"])
                exploration_drinks = int(values["exploration_drinks"])
                brain_verb = "fue decisión" if brain_drinks == 1 else "fueron decisiones"
                exploration_verb = "ocurrió" if exploration_drinks == 1 else "ocurrieron"
                descriptions.append(
                    f"{label} lograron beber {successful_drinks} {drink_word}; esas acciones sí redujeron la sed "
                    f"y entraron al replay como experiencias positivas. {brain_drinks} {brain_verb} del brain "
                    f"y {exploration_drinks} {exploration_verb} durante exploración aleatoria."
                )
            else:
                descriptions.append(
                    f"{label} no lograron beber ni una vez, así que todavía no existe evidencia de que "
                    "hayan aprendido a buscar agua."
                )
        if self.termination_reason == "human_extinction":
            descriptions.append(
                f"Los humanos todavía no aprendieron a sobrevivir todo el run: el último murió en el tick {self.current_tick}."
            )
        if comparison is not None:
            tick_change = int(comparison["last_human_tick_change"])
            mean_change = float(comparison["mean_human_survival_change"])
            if tick_change > 0 and mean_change > 0:
                descriptions.append(
                    "Frente al run anterior, hubo una mejora: el último humano vivió "
                    f"{tick_change} ticks más y la supervivencia humana media cambió "
                    f"{comparison['previous_mean_human_survival']:.1f} -> "
                    f"{comparison['current_mean_human_survival']:.1f} ticks."
                )
            elif tick_change < 0 or mean_change < 0:
                descriptions.append(
                    "Frente al run anterior no hubo una mejora consistente: el último humano cambió "
                    f"de {comparison['previous_last_human_tick']} a "
                    f"{comparison['current_last_human_tick']} ticks y la media cambió "
                    f"{comparison['previous_mean_human_survival']:.1f} -> "
                    f"{comparison['current_mean_human_survival']:.1f}."
                )
            else:
                descriptions.append("La supervivencia humana quedó prácticamente igual al run anterior.")
        descriptions.append(
            "Sabremos si el aprendizaje ayuda cuando los siguientes runs, usando estos mismos pesos, "
            "aumenten la supervivencia de forma repetida en varios seeds."
        )
        return descriptions

    def _load_previous_run_comparison(
        self, by_type: dict[str, dict[str, float | str]]
    ) -> dict[str, float | int] | None:
        if self.source_checkpoint is None:
            return None
        previous_summary_path = (
            self.root
            / "results"
            / self.source_checkpoint.parent.name
            / self.source_checkpoint.name
            / "run_summary.json"
        )
        if not previous_summary_path.is_file():
            return None
        try:
            previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
            previous_tick = int(previous["ticks_executed"])
            previous_configured_ticks = int(
                previous.get("configured_ticks", previous_tick)
            )
            if previous_configured_ticks != self.config.num_ticks:
                return None
            previous_mean = float(previous["by_type"]["human"]["mean_survival"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        current_mean = float(by_type["human"]["mean_survival"])
        return {
            "previous_run_number": int(previous["run_number"]),
            "previous_last_human_tick": previous_tick,
            "current_last_human_tick": self.current_tick,
            "last_human_tick_change": self.current_tick - previous_tick,
            "previous_mean_human_survival": previous_mean,
            "current_mean_human_survival": current_mean,
            "mean_human_survival_change": current_mean - previous_mean,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _percent_change(first: float, last: float) -> float:
    return 0.0 if abs(first) < 1e-12 else (last - first) / abs(first) * 100.0


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _friendly_action(action: str) -> str:
    return {
        "MOVE_UP": "moverse hacia arriba",
        "MOVE_DOWN": "moverse hacia abajo",
        "MOVE_LEFT": "moverse hacia la izquierda",
        "MOVE_RIGHT": "moverse hacia la derecha",
        "EAT": "intentar comer",
        "DRINK": "intentar beber",
        "REST": "descansar",
        "WAIT": "esperar",
    }.get(action, action)


def _mean_agent_epsilon(agents: list[object]) -> float:
    values = [float(getattr(agent, "last_epsilon", 0.0)) for agent in agents]
    return sum(values) / len(values) if values else 0.0

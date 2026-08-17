from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from agents.base_agent import BaseAgent
from world.world import World


@dataclass
class MetricsRecorder:
    agent_rows: list[dict[str, object]] = field(default_factory=list)
    summary_rows: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        tick: int,
        world: World,
        tick_rewards: dict[str, float],
        tick_losses: dict[str, float | None],
        tick_actions: dict[str, str],
        tick_components: dict[str, dict[str, float]] | None = None,
        tick_events: dict[str, dict[str, object]] | None = None,
    ) -> None:
        tick_components = tick_components or {}
        tick_events = tick_events or {}
        for agent in world.agents:
            components = tick_components.get(agent.id, {})
            event = tick_events.get(agent.id, {})
            self.agent_rows.append(
                {
                    "tick": tick,
                    "agent_id": agent.id,
                    "agent_type": agent.agent_type,
                    "alive": agent.alive,
                    "health": agent.health,
                    "energy": agent.energy,
                    "hunger": agent.hunger,
                    "thirst": agent.thirst,
                    "reward": tick_rewards.get(agent.id, 0.0),
                    "cumulative_reward": agent.total_reward,
                    "loss": tick_losses.get(agent.id),
                    "action": tick_actions.get(agent.id, "DEAD"),
                    "ate": "ate_while_hungry" in components,
                    "drank": "drank_while_thirsty" in components,
                    "resource_progress": components.get("resource_progress", 0.0),
                    "exploration": event.get("exploration"),
                    "epsilon": event.get("epsilon"),
                    "exploration_profile": agent.exploration_profile,
                    "personal_replay_size": len(agent.trainer.replay_buffer),
                    "horde_replay_size": len(agent.trainer.learning_replay_buffer),
                    "trained": event.get("trained", False),
                    "training_steps": event.get(
                        "training_steps", agent.trainer.training_steps
                    ),
                    "x": agent.x,
                    "y": agent.y,
                }
            )
        humans = [agent for agent in world.agents if agent.agent_type == "human"]
        animals = [agent for agent in world.agents if agent.agent_type == "animal"]
        losses = [loss for loss in tick_losses.values() if loss is not None]
        self.summary_rows.append(
            {
                "tick": tick,
                "living_humans": sum(agent.alive for agent in humans),
                "living_animals": sum(agent.alive for agent in animals),
                "average_human_reward": _mean(agent.total_reward for agent in humans),
                "average_animal_reward": _mean(agent.total_reward for agent in animals),
                "average_loss": _mean(losses) if losses else None,
                "food_remaining": len(world.food),
                "water_remaining": len(world.water),
                "deaths": sum(not agent.alive for agent in world.agents),
                "human_horde_replay": _horde_size(humans),
                "animal_horde_replay": _horde_size(animals),
            }
        )

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        agents_path = output_dir / "agents.csv"
        summary_path = output_dir / "summary.csv"
        pd.DataFrame(self.agent_rows).to_csv(agents_path, index=False)
        pd.DataFrame(self.summary_rows).to_csv(summary_path, index=False)
        return agents_path, summary_path

    def agent_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.agent_rows)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary_rows)


def _mean(values: object) -> float:
    materialized = list(values)  # type: ignore[arg-type]
    return float(sum(materialized) / len(materialized)) if materialized else 0.0


def _horde_size(agents: list[BaseAgent]) -> int:
    if not agents:
        return 0
    return len(agents[0].trainer.learning_replay_buffer)

"""Agent state, perception, action selection, and biology."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from agents.brain import AgentBrain
from agents.spatial_memory import SpatialMemory
from learning.trainer import DQNTrainer

if TYPE_CHECKING:
    from config import BrainConfig, SimulationConfig
    from world.world import World


@dataclass
class BaseAgent:
    id: str
    x: int
    y: int
    config: "SimulationConfig" = field(repr=False)
    brain_config: "BrainConfig" = field(repr=False)
    input_size: int
    agent_type: str
    age: int = 0
    health: float = 1.0
    energy: float = 1.0
    hunger: float = 0.0
    thirst: float = 0.0
    alive: bool = True
    total_reward: float = 0.0
    steps_survived: int = 0

    def __post_init__(self) -> None:
        from world.grid import Action

        self.brain = AgentBrain(
            self.input_size,
            self.brain_config.hidden_sizes,
            len(Action),
            need_input_size=type(self).NEED_INPUT_SIZE,
        )
        self.trainer = DQNTrainer(self.brain, self.brain_config)
        self.action_counts: dict[str, int] = {action.name: 0 for action in Action}
        self.recent_positions: list[tuple[int, int]] = []
        self.exploration_profile, self.exploration_epsilon = self._exploration_profile()
        self.last_epsilon = self.exploration_epsilon
        self.last_was_exploration = True
        self.last_q_values: list[float] = [0.0 for _ in Action]
        self.decision_steps = 0
        self.unnecessary_need_streaks = {"EAT": 0, "DRINK": 0, "REST": 0}
        self.spatial_memory = SpatialMemory()
        self.last_brain_activations: dict[str, list[float]] = {}

    def perceive(self, world: "World") -> torch.Tensor:
        raise NotImplementedError

    def choose_action(self, state: torch.Tensor, global_step: int | None = None) -> int:
        """epsilon-greedy: the non-random branch is the brain forward pass."""
        from world.grid import Action

        epsilon = self.exploration_epsilon
        # We calculate Q-values on every tick, even during random exploration.
        # This lets the web laboratory show what the brain currently predicts.
        self.brain.eval()
        with torch.no_grad():
            q_batch, activation_batch = self.brain.forward_with_activations(
                state.unsqueeze(0)
            )
            q_values = q_batch.squeeze(0)
        exploring = random.random() < epsilon
        if exploring:
            action_index = random.randrange(len(Action))
        else:
            action_index = int(q_values.argmax().item())
        if not 0 <= action_index < len(Action):
            raise RuntimeError(f"Brain selected invalid action {action_index}")
        self.last_epsilon = epsilon
        self.last_was_exploration = exploring
        self.last_q_values = [float(value) for value in q_values.tolist()]
        self.last_brain_activations = {
            name: [float(value) for value in tensor.squeeze(0).tolist()]
            for name, tensor in activation_batch.items()
        }
        self.decision_steps += 1
        self.action_counts[Action(action_index).name] += 1
        return action_index

    def _exploration_profile(self) -> tuple[str, float]:
        """Assign a reproducible individual exploration tendency within its species."""
        try:
            index = max(1, int(self.id.rsplit("_", 1)[-1]))
        except ValueError:
            index = 1
        population = (
            self.config.num_humans
            if self.agent_type == "human"
            else self.config.num_animals
        )
        scout_count = max(1, math.ceil(population * self.config.epsilon_scout_fraction))
        if index <= scout_count:
            return "scout", self.config.epsilon_scout
        normal_count = max(1, population - scout_count)
        normal_rank = index - scout_count - 1
        fraction = normal_rank / max(1, normal_count - 1)
        epsilon = self.config.epsilon_standard_min + fraction * (
            self.config.epsilon_standard_max - self.config.epsilon_standard_min
        )
        return "standard", epsilon

    def update_biology(self, rested: bool = False) -> tuple[float, bool]:
        """Advance needs and return (health_delta, died_this_tick)."""
        old_health = self.health
        self.age += 1
        self.steps_survived += 1
        self.hunger = min(1.0, self.hunger + self.config.hunger_per_tick)
        self.thirst = min(1.0, self.thirst + self.config.thirst_per_tick)
        energy_change = self.config.rest_energy_gain if rested else -self.config.energy_per_tick
        self.energy = min(1.0, max(0.0, self.energy + energy_change))
        if self.hunger >= 1.0:
            self.health -= self.config.starvation_damage
        if self.thirst >= 1.0:
            self.health -= self.config.dehydration_damage
        if self.energy <= 0.0:
            self.health -= self.config.starvation_damage / 2
        self.health = max(0.0, min(1.0, self.health))
        died = self.health <= 0.0
        if died:
            self.alive = False
        return self.health - old_health, died

    def remember_and_learn(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> float | None:
        self.trainer.remember(state, action, reward, next_state, done)
        return self.trainer.train_step()

    def note_position(self) -> None:
        self.recent_positions.append((self.x, self.y))
        self.recent_positions = self.recent_positions[-4:]

    def unnecessary_need_penalty(self, action_name: str, unnecessary: bool) -> float:
        """Increase the cost of repeatedly satisfying a need that is not urgent."""
        if action_name not in self.unnecessary_need_streaks:
            return 0.0
        if not unnecessary:
            self.unnecessary_need_streaks[action_name] = 0
            return 0.0
        self.unnecessary_need_streaks[action_name] += 1
        return min(
            self.config.unnecessary_action_penalty_cap,
            self.config.unnecessary_action_penalty_base
            * self.unnecessary_need_streaks[action_name],
        )

    @property
    def is_repeating_position(self) -> bool:
        return len(self.recent_positions) == 4 and len(set(self.recent_positions)) <= 2

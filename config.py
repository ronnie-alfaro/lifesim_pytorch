"""Central configuration for LifeSim experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BrainConfig:
    hidden_sizes: list[int]
    learning_rate: float = 0.001
    replay_capacity: int = 5_000
    batch_size: int = 32
    # Longer horizon: consequences such as starvation must affect earlier choices.
    gamma: float = 0.99
    train_every: int = 4
    target_update_interval: int = 100
    positive_replay_fraction: float = 0.25


@dataclass(slots=True)
class SimulationConfig:
    reward_version: int = 5
    brain_architecture_version: int = 2
    # A denser matrix gives the agents more room without making the web panel larger.
    grid_width: int = 60
    grid_height: int = 40
    num_humans: int = 5
    num_animals: int = 10
    initial_food: int = 90
    initial_water: int = 120
    initial_obstacles: int = 100
    food_cluster_count: int = 9
    water_cluster_count: int = 9
    max_food: int = 130
    food_respawn_probability: float = 0.08
    # Kept in serialized configs for backwards compatibility. Water is permanent.
    water_respawn_probability: float = 0.0
    num_ticks: int = 5_000
    stop_when_no_humans: bool = True
    status_every: int = 100
    render_every: int = 0
    compact_console: bool = False
    generate_plots: bool = True
    max_health: float = 1.0
    max_energy: float = 1.0
    # The map doubled from 30x20 to 60x40, so metabolism is scaled to preserve
    # the original need-per-distance budget at one movement cell per tick.
    hunger_per_tick: float = 0.005
    thirst_per_tick: float = 0.005
    energy_per_tick: float = 0.005
    starvation_damage: float = 0.025
    dehydration_damage: float = 0.020
    rest_energy_gain: float = 0.10
    food_hunger_reduction: float = 0.50
    water_thirst_reduction: float = 0.60
    need_action_threshold: float = 0.25
    # Plan before needs become urgent: safe <= 50%, danger begins at 70%.
    priority_need_threshold: float = 0.25
    need_reward_scale: float = 1.0
    unnecessary_action_penalty_base: float = 0.10
    unnecessary_action_penalty_cap: float = 1.0
    survival_priority_threshold: float = 0.25
    survival_priority_penalty_base: float = 0.05
    survival_priority_penalty_scale: float = 0.25
    survival_priority_penalty_cap: float = 0.60
    need_safe_target: float = 0.50
    need_danger_threshold: float = 0.70
    need_penalty_at_danger: float = 0.30
    need_penalty_cap: float = 0.80
    need_safe_recovery_reward: float = 0.25
    survival_travel_reserve_ticks: float = 5.0
    vision_radius: int = 6
    # Resource scent is longer-range than visual detail so a safe route is
    # usually discoverable before hunger/thirst reaches 70%.
    resource_sense_radius: int = 100
    spatial_memory_max_age: int = 200
    # Horde exploration profiles: most individuals exploit heavily, while a
    # small stable minority remains curious and supplies novel group experience.
    epsilon_standard_min: float = 0.01
    epsilon_standard_max: float = 0.15
    epsilon_scout: float = 0.50
    epsilon_scout_fraction: float = 0.10
    horde_learning_enabled: bool = True
    horde_replay_capacity: int = 20_000
    debug_rewards: bool = False
    human_brain: BrainConfig = field(
        # need encoder, spatial encoder, fused decision layer
        default_factory=lambda: BrainConfig(hidden_sizes=[16, 32, 32])
    )
    animal_brain: BrainConfig = field(
        default_factory=lambda: BrainConfig(hidden_sizes=[12, 24, 24])
    )

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.epsilon_standard_min
            <= self.epsilon_standard_max
            <= self.epsilon_scout
            <= 0.50
        ):
            raise ValueError(
                "Epsilon profiles must satisfy 0 <= normal min <= normal max "
                "<= scout <= 0.50"
            )
        if not 0.0 < self.epsilon_scout_fraction <= 1.0:
            raise ValueError("epsilon_scout_fraction must be in (0, 1]")
        if self.horde_replay_capacity <= 0:
            raise ValueError("horde_replay_capacity must be positive")
        if not 0.0 < self.priority_need_threshold < self.need_safe_target < self.need_danger_threshold < 1.0:
            raise ValueError(
                "Need thresholds must satisfy priority < safe target < danger < 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardResult:
    total: float
    components: dict[str, float]


def calculate_reward(
    *, ate: bool, drank: bool, invalid: bool, reached_needed_resource: bool,
    health_delta: float, died: bool, repeating: bool, resource_progress: float = 0.0,
    rested: bool = False, need_reward: float = 0.0,
    unnecessary_action_penalty: float = 0.0,
) -> RewardResult:
    """Small, inspectable reward vocabulary; no action policy is encoded here."""
    components: dict[str, float] = {"survival": 0.01}
    if ate:
        components["ate_while_hungry"] = need_reward
    if drank:
        components["drank_while_thirsty"] = need_reward
    if rested and need_reward > 0.0:
        components["rested_while_tired"] = need_reward
    if unnecessary_action_penalty > 0.0:
        components["unnecessary_need_action"] = -unnecessary_action_penalty
    if reached_needed_resource:
        components["reached_needed_resource"] = 0.2
    if resource_progress:
        components["resource_progress"] = resource_progress
    if health_delta > 0:
        components["health_recovered"] = 0.2
    if invalid:
        components["invalid_action"] = -0.1
    if repeating:
        components["repetitive_movement"] = -0.02
    if died:
        components["death"] = -5.0
        components.pop("survival", None)
    return RewardResult(sum(components.values()), components)

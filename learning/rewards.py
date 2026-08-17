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
    survival_priority_penalty: float = 0.0,
    need_safety_signal: float = 0.0,
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
    if survival_priority_penalty > 0.0:
        components["ignored_survival_priority"] = -survival_priority_penalty
    if need_safety_signal > 0.0:
        components["returned_to_safe_need_zone"] = need_safety_signal
    elif need_safety_signal < 0.0:
        components["unsafe_need_level"] = need_safety_signal
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


def calculate_survival_priority_penalty(
    *,
    action_name: str,
    hunger: float,
    thirst: float,
    energy: float,
    health: float,
    threshold: float,
    base: float,
    scale: float,
    cap: float,
    movement_has_known_target: bool = False,
    resource_progress: float = 0.0,
) -> float:
    """Penalize passive or irrelevant choices while a vital need is urgent.

    Movement remains available because it may be a search action. EAT, DRINK,
    and REST are considered correct only when their corresponding need is
    already urgent. The penalty grows with urgency and low health.
    """
    needs_by_action = {
        "EAT": hunger,
        "DRINK": thirst,
        "REST": 1.0 - energy,
    }
    urgency = max(needs_by_action.values())
    if urgency < threshold:
        return 0.0
    if action_name.startswith("MOVE_"):
        # Searching without a memory remains valid. Once a target is known,
        # moving closer is the survival action and wandering is not.
        if not movement_has_known_target or resource_progress > 0.0:
            return 0.0
    if action_name in needs_by_action and needs_by_action[action_name] >= threshold:
        return 0.0
    severity = (urgency - threshold) / max(1e-9, 1.0 - threshold)
    health_multiplier = 1.0 + (1.0 - health)
    return min(cap, (base + scale * severity) * health_multiplier)


def calculate_need_safety_signal(
    *,
    hunger_before: float,
    thirst_before: float,
    hunger_after: float,
    thirst_after: float,
    safe_target: float,
    danger_threshold: float,
    penalty_at_danger: float,
    penalty_cap: float,
    recovery_reward: float,
) -> float:
    """Dense outcome signal that teaches agents to remain below 50%.

    This does not choose an action or clamp physiology. It rewards actually
    returning a need to the safe zone and charges an increasingly steep cost
    for each tick spent between the safe target, danger and terminal levels.
    """
    recovered = sum(
        before > safe_target and after <= safe_target
        for before, after in (
            (hunger_before, hunger_after),
            (thirst_before, thirst_after),
        )
    )
    def level_cost(value: float) -> float:
        if value <= safe_target:
            return 0.0
        if value <= danger_threshold:
            fraction = (value - safe_target) / (danger_threshold - safe_target)
            return penalty_at_danger * fraction * fraction
        fraction = (value - danger_threshold) / (1.0 - danger_threshold)
        return penalty_at_danger + (penalty_cap - penalty_at_danger) * fraction

    cost = min(
        penalty_cap,
        level_cost(hunger_after) + level_cost(thirst_after),
    )
    return recovery_reward * recovered - cost

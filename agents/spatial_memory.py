"""Explicit, inspectable episodic memory for resource locations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from world.world import World


@dataclass
class ResourceMemory:
    position: tuple[int, int] | None = None
    last_seen_age: int | None = None


class SpatialMemory:
    """Remember last-seen food and water; the map itself is not inherited."""

    def __init__(self) -> None:
        self.food = ResourceMemory()
        self.water = ResourceMemory()

    def observe(self, agent: "BaseAgent", world: "World") -> None:
        for resource in ("food", "water"):
            memory = self.food if resource == "food" else self.water
            visible = world.visible_resources(
                agent.x, agent.y, resource, agent.config.resource_sense_radius
            )
            need = agent.hunger if resource == "food" else agent.thirst
            if (
                memory.position in visible
                and need >= agent.config.priority_need_threshold
            ):
                # Commit to a living target instead of chasing whichever cell
                # becomes one step nearer while this need is a priority.
                memory.last_seen_age = agent.age
                continue
            if visible:
                memory.position = min(
                    visible,
                    key=lambda position: abs(position[0] - agent.x)
                    + abs(position[1] - agent.y),
                )
                memory.last_seen_age = agent.age
            elif (
                memory.position is not None
                and world.manhattan_distance((agent.x, agent.y), memory.position)
                <= agent.config.resource_sense_radius
            ):
                # The remembered food was consumed or changed before arrival.
                memory.position = None
                memory.last_seen_age = None

    def features(
        self,
        resource: Literal["food", "water"],
        agent: "BaseAgent",
        world: "World",
    ) -> list[float]:
        memory = self.food if resource == "food" else self.water
        if memory.position is None or memory.last_seen_age is None:
            return [0.0, 0.0, 0.0, 1.0]
        age = max(0, agent.age - memory.last_seen_age)
        max_age = max(1, agent.config.spatial_memory_max_age)
        confidence = max(0.0, 1.0 - age / max_age)
        dx = (memory.position[0] - agent.x) / max(1, world.width - 1)
        dy = (memory.position[1] - agent.y) / max(1, world.height - 1)
        return [dx, dy, confidence, min(1.0, age / max_age)]

    def snapshot(self, agent: "BaseAgent") -> dict[str, object]:
        return {
            name: self._resource_snapshot(memory, agent)
            for name, memory in (("food", self.food), ("water", self.water))
        }

    @staticmethod
    def _resource_snapshot(
        memory: ResourceMemory, agent: "BaseAgent"
    ) -> dict[str, object]:
        age = (
            None
            if memory.last_seen_age is None
            else max(0, agent.age - memory.last_seen_age)
        )
        confidence = (
            0.0
            if age is None
            else max(0.0, 1.0 - age / max(1, agent.config.spatial_memory_max_age))
        )
        return {
            "position": list(memory.position) if memory.position is not None else None,
            "age": age,
            "confidence": confidence,
        }

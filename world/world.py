from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agents.animal import Animal
from agents.human import Human
from config import SimulationConfig
from world.grid import MOVEMENT_DELTAS, Action

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


@dataclass(frozen=True)
class ActionResult:
    action: Action
    invalid: bool = False
    ate: bool = False
    drank: bool = False
    rested: bool = False
    reached_needed_resource: bool = False
    need: Literal["hunger", "thirst", "energy"] | None = None
    necessity: float = 0.0
    unnecessary_need_action: bool = False


class World:
    def __init__(self, config: SimulationConfig, populate: bool = True) -> None:
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        if self.width <= 1 or self.height <= 1:
            raise ValueError("World dimensions must both exceed one")
        self.food: set[tuple[int, int]] = set()
        self.water: set[tuple[int, int]] = set()
        self.obstacles: set[tuple[int, int]] = set()
        self.agents: list[BaseAgent] = []
        if populate:
            self._populate()

    def _empty_position(self, include_agents: bool = True) -> tuple[int, int]:
        occupied = self.food | self.water | self.obstacles
        if include_agents:
            occupied |= {(agent.x, agent.y) for agent in self.agents}
        if len(occupied) >= self.width * self.height:
            raise RuntimeError("World has no free positions")
        while True:
            position = (random.randrange(self.width), random.randrange(self.height))
            if position not in occupied:
                return position

    def _populate(self) -> None:
        for _ in range(self.config.initial_obstacles):
            self.obstacles.add(self._empty_position(False))
        self._populate_resource_clusters(
            self.food, self.config.initial_food, self.config.food_cluster_count
        )
        self._populate_resource_clusters(
            self.water, self.config.initial_water, self.config.water_cluster_count
        )
        for index in range(1, self.config.num_humans + 1):
            x, y = self._empty_position()
            self.agents.append(Human.create(f"human_{index:03d}", x, y, self.config))
        for index in range(1, self.config.num_animals + 1):
            x, y = self._empty_position()
            self.agents.append(Animal.create(f"animal_{index:03d}", x, y, self.config))

    @property
    def living_agents(self) -> list[BaseAgent]:
        return [agent for agent in self.agents if agent.alive]

    def update_resources(self) -> None:
        """Grow food beside existing plants; permanent water never depletes."""
        if (
            len(self.food) < self.config.max_food
            and random.random() < self.config.food_respawn_probability
        ):
            self._grow_resource_cluster(self.food)

    def _populate_resource_clusters(
        self,
        resource: set[tuple[int, int]],
        count: int,
        cluster_count: int,
    ) -> None:
        """Populate an exact resource count as a small number of contiguous patches."""
        if count <= 0:
            return
        seeds = min(count, max(1, cluster_count))
        for _ in range(seeds):
            resource.add(self._empty_position(False))
        failed_growth_attempts = 0
        while len(resource) < count:
            if self._grow_resource_cluster(resource):
                failed_growth_attempts = 0
                continue
            failed_growth_attempts += 1
            if failed_growth_attempts >= 4:
                resource.add(self._empty_position(False))
                failed_growth_attempts = 0

    def _grow_resource_cluster(self, resource: set[tuple[int, int]]) -> bool:
        """Add one cardinal neighbour, returning False only if no edge is free."""
        if not resource:
            resource.add(self._empty_position(False))
            return True
        positions = list(resource)
        random.shuffle(positions)
        directions = list(MOVEMENT_DELTAS.values())
        for x, y in positions:
            random.shuffle(directions)
            for dx, dy in directions:
                candidate = (x + dx, y + dy)
                if self._resource_position_is_free(candidate):
                    resource.add(candidate)
                    return True
        return False

    def _resource_position_is_free(self, position: tuple[int, int]) -> bool:
        return (
            self.in_bounds(*position)
            and position not in self.food
            and position not in self.water
            and position not in self.obstacles
        )

    def execute_action(self, agent: BaseAgent, action_index: int) -> ActionResult:
        try:
            action = Action(action_index)
        except ValueError as error:
            raise ValueError(f"Action index outside valid range: {action_index}") from error
        if not agent.alive:
            return ActionResult(action, invalid=True)
        if action in MOVEMENT_DELTAS:
            dx, dy = MOVEMENT_DELTAS[action]
            new_position = (agent.x + dx, agent.y + dy)
            if not self.in_bounds(*new_position) or new_position in self.obstacles:
                return ActionResult(action, invalid=True)
            agent.x, agent.y = new_position
            needed = (new_position in self.food and agent.hunger >= 0.5) or (
                new_position in self.water and agent.thirst >= 0.5
            )
            return ActionResult(action, reached_needed_resource=needed)
        if action is Action.EAT:
            position = self.resource_position_in_reach(agent, "food")
            if position is None:
                return ActionResult(action, invalid=True)
            hunger_before = agent.hunger
            self.food.remove(position)
            agent.hunger = max(0.0, agent.hunger - self.config.food_hunger_reduction)
            necessary = hunger_before >= self.config.need_action_threshold
            return ActionResult(
                action,
                ate=necessary,
                need="hunger",
                necessity=hunger_before if necessary else 0.0,
                unnecessary_need_action=not necessary,
            )
        if action is Action.DRINK:
            position = self.resource_position_in_reach(agent, "water")
            if position is None:
                return ActionResult(action, invalid=True)
            thirst_before = agent.thirst
            # Water represents a permanent source (pond/river), not one drink.
            agent.thirst = max(0.0, agent.thirst - self.config.water_thirst_reduction)
            necessary = thirst_before >= self.config.need_action_threshold
            return ActionResult(
                action,
                drank=necessary,
                need="thirst",
                necessity=thirst_before if necessary else 0.0,
                unnecessary_need_action=not necessary,
            )
        if action is Action.REST:
            energy_need = 1.0 - agent.energy
            necessary = energy_need >= self.config.need_action_threshold
            return ActionResult(
                action,
                rested=True,
                need="energy",
                necessity=energy_need if necessary else 0.0,
                unnecessary_need_action=not necessary,
            )
        return ActionResult(action)

    def resource_position_in_reach(
        self, agent: BaseAgent, resource: Literal["food", "water"]
    ) -> tuple[int, int] | None:
        """Agents can consume a resource on their cell or a cardinal neighbour."""
        positions = self.food if resource == "food" else self.water
        candidates = (
            (agent.x, agent.y),
            (agent.x, agent.y - 1),
            (agent.x, agent.y + 1),
            (agent.x - 1, agent.y),
            (agent.x + 1, agent.y),
        )
        return next((position for position in candidates if position in positions), None)

    def resource_in_reach(
        self, agent: BaseAgent, resource: Literal["food", "water"]
    ) -> bool:
        return self.resource_position_in_reach(agent, resource) is not None

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    @staticmethod
    def manhattan_distance(
        first: tuple[int, int], second: tuple[int, int]
    ) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def visible_resources(
        self,
        x: int,
        y: int,
        resource: Literal["food", "water"],
        radius: int,
    ) -> list[tuple[int, int]]:
        positions = self.food if resource == "food" else self.water
        return [
            position
            for position in positions
            if self.manhattan_distance((x, y), position) <= radius
        ]

    def cardinal_obstacle_flags(self, x: int, y: int) -> list[float]:
        """Blocked/out-of-bounds flags ordered up, down, left, right."""
        neighbours = ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        return [
            float(not self.in_bounds(*position) or position in self.obstacles)
            for position in neighbours
        ]

    def normalized_direction_to_nearest(
        self, x: int, y: int, resource: Literal["food", "water"]
    ) -> tuple[float, float]:
        positions = self.food if resource == "food" else self.water
        if not positions:
            return 0.0, 0.0
        target_x, target_y = min(positions, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        return (
            (target_x - x) / max(1, self.width - 1),
            (target_y - y) / max(1, self.height - 1),
        )

    def distance_to_nearest(
        self, x: int, y: int, resource: Literal["food", "water"]
    ) -> int | None:
        positions = self.food if resource == "food" else self.water
        if not positions:
            return None
        return min(abs(resource_x - x) + abs(resource_y - y) for resource_x, resource_y in positions)

    def normalized_agent_distance(self, source: BaseAgent, agent_type: str) -> float:
        others = [
            agent for agent in self.living_agents
            if agent.id != source.id and agent.agent_type == agent_type
        ]
        if not others:
            return 1.0
        distance = min(math.hypot(agent.x - source.x, agent.y - source.y) for agent in others)
        diagonal = math.hypot(self.width - 1, self.height - 1)
        return min(1.0, distance / diagonal)

    def validate(self) -> None:
        for agent in self.agents:
            if not self.in_bounds(agent.x, agent.y):
                raise RuntimeError(f"Agent {agent.id} escaped the grid at {(agent.x, agent.y)}")
        if self.obstacles & self.food or self.obstacles & self.water:
            raise RuntimeError("A resource overlaps an obstacle")

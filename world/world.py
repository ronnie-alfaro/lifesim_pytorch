from __future__ import annotations

import math
import random
from collections import deque
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
    attacked: bool = False
    attack_damage: float = 0.0
    killed: bool = False
    target_id: str | None = None
    gathered: bool = False
    deposited: bool = False
    fed_baby: bool = False
    mated: bool = False


@dataclass
class Stockpile:
    agent_type: str
    x: int
    y: int
    food: int = 0

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class ActionConstraints:
    mask: list[bool]
    mode: str
    priority: str | None = None
    target: tuple[int, int] | None = None


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
        self.stockpiles: dict[str, Stockpile] = {}
        self._distance_map_cache: dict[
            tuple[int, int], dict[tuple[int, int], int]
        ] = {}
        if populate:
            self._populate()

    def _empty_position(self, include_agents: bool = True) -> tuple[int, int]:
        occupied = self.food | self.water | self.obstacles | {
            stockpile.position for stockpile in self.stockpiles.values()
        }
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
        # Food uses one maximally separated seed per cell. Water remains in
        # contiguous sources, but edible cells are scattered across the map.
        self._populate_distributed_resource(self.food, self.food_target)
        self._populate_resource_clusters(
            self.water, self.config.initial_water, self.config.water_cluster_count
        )
        for index in range(1, self.config.num_humans + 1):
            x, y = self._empty_position()
            self.agents.append(Human.create(f"human_{index:03d}", x, y, self.config))
        for index in range(1, self.config.num_animals + 1):
            x, y = self._empty_position()
            self.agents.append(Animal.create(f"animal_{index:03d}", x, y, self.config))
        self.initialize_stockpiles()

    def initialize_stockpiles(self) -> None:
        """Anchor one communal reserve per species at its first living founder."""
        self.stockpiles = {}
        for agent_type in ("human", "animal"):
            founder = next(
                (agent for agent in self.agents if agent.agent_type == agent_type), None
            )
            if founder is not None:
                self.stockpiles[agent_type] = Stockpile(
                    agent_type, founder.x, founder.y
                )

    def random_empty_positions(self, count: int) -> list[tuple[int, int]]:
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.food
            and (x, y) not in self.water
            and (x, y) not in self.obstacles
        ]
        if count > len(candidates):
            raise RuntimeError(f"World has room for only {len(candidates)} agents")
        return random.sample(candidates, count)

    @property
    def living_agents(self) -> list[BaseAgent]:
        return [agent for agent in self.agents if agent.alive]

    def update_resources(self) -> None:
        """Gradually replenish missing food, preserving temporary scarcity."""
        stored = sum(stockpile.food for stockpile in self.stockpiles.values())
        carried = sum(agent.carried_food for agent in self.living_agents)
        missing = max(0, self.food_target - len(self.food) - stored - carried)
        replacements = sum(
            random.random() < self.config.food_respawn_probability
            for _ in range(missing)
        )
        if replacements:
            self._populate_distributed_resource(
                self.food, min(self.food_target, len(self.food) + replacements)
            )

    @property
    def food_target(self) -> int:
        inhabitants = len(self.living_agents) if self.agents else (
            self.config.num_humans + self.config.num_animals
        )
        return max(self.config.initial_food, inhabitants * self.config.food_per_agent)

    @property
    def total_food_supply(self) -> int:
        return (
            len(self.food)
            + sum(stockpile.food for stockpile in self.stockpiles.values())
            + sum(agent.carried_food for agent in self.living_agents)
        )

    def advance_social_dynamics(self) -> list[BaseAgent]:
        """Advance hearts, pregnancies and dependent babies by one tick."""
        living_ids = {agent.id for agent in self.living_agents}
        for mother in self.agents:
            mother.dependent_ids = [
                child_id for child_id in mother.dependent_ids if child_id in living_ids
            ]

        newly_pregnant: set[str] = set()
        for agent in self.living_agents:
            if agent.heart_ticks_remaining <= 0:
                continue
            agent.heart_ticks_remaining -= 1
            if agent.heart_ticks_remaining == 0:
                if agent.sex == "F":
                    agent.pregnant_by_id = agent.heart_partner_id
                    agent.pregnancy_ticks_remaining = self.config.pregnancy_ticks
                    newly_pregnant.add(agent.id)
                agent.heart_partner_id = None

        newborns: list[BaseAgent] = []
        for mother in list(self.living_agents):
            if mother.pregnancy_ticks_remaining <= 0 or mother.id in newly_pregnant:
                continue
            mother.pregnancy_ticks_remaining -= 1
            if mother.pregnancy_ticks_remaining == 0:
                newborns.extend(self._give_birth(mother))
                mother.pregnant_by_id = None

        newborn_ids = {baby.id for baby in newborns}
        agents_by_id = {agent.id: agent for agent in self.agents}
        for baby in self.living_agents:
            if baby.dependent_ticks_remaining <= 0 or baby.id in newborn_ids:
                continue
            mother = agents_by_id.get(baby.mother_id or "")
            if mother is None or not mother.alive:
                baby.dependent_ticks_remaining = 0
                baby.mother_id = None
                continue
            self._follow_mother(baby, mother)
            baby.dependent_ticks_remaining -= 1
            if baby.dependent_ticks_remaining == 0:
                baby.mother_id = None
                mother.dependent_ids = [
                    child_id for child_id in mother.dependent_ids if child_id != baby.id
                ]

        return newborns

    def _give_birth(self, mother: BaseAgent) -> list[BaseAgent]:
        roll = random.random()
        litter_size = 1 if roll < 0.70 else 2 if roll < 0.95 else 3
        existing_numbers = [
            int(agent.id.rsplit("_", 1)[-1])
            for agent in self.agents
            if agent.agent_type == mother.agent_type
            and agent.id.rsplit("_", 1)[-1].isdigit()
        ]
        next_number = max(existing_numbers, default=0) + 1
        agent_class = Human if mother.agent_type == "human" else Animal
        babies: list[BaseAgent] = []
        for offset in range(litter_size):
            baby_id = f"{mother.agent_type}_{next_number + offset:03d}"
            baby = agent_class.create(baby_id, mother.x, mother.y, self.config)
            if baby.agent_type == "animal":
                baby.predator = mother.predator
            baby.mother_id = mother.id
            baby.dependent_ticks_remaining = self.config.dependent_baby_ticks
            baby.hunger = self.config.need_action_threshold + 0.10
            babies.append(baby)
        self.agents.extend(babies)
        mother.dependent_ids.extend(baby.id for baby in babies)
        mother.children_born += len(babies)
        return babies

    def _follow_mother(self, baby: BaseAgent, mother: BaseAgent) -> None:
        if (baby.x, baby.y) == (mother.x, mother.y):
            return
        dx = 0 if baby.x == mother.x else (1 if mother.x > baby.x else -1)
        dy = 0 if baby.y == mother.y else (1 if mother.y > baby.y else -1)
        candidates = [(baby.x + dx, baby.y), (baby.x, baby.y + dy)]
        for position in candidates:
            if self.in_bounds(*position) and position not in self.obstacles:
                baby.x, baby.y = position
                return

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
            resource.add(self._spread_resource_seed(resource))
        failed_growth_attempts = 0
        while len(resource) < count:
            if self._grow_resource_cluster(resource):
                failed_growth_attempts = 0
                continue
            failed_growth_attempts += 1
            if failed_growth_attempts >= 4:
                resource.add(self._empty_position(False))
                failed_growth_attempts = 0

    def _populate_distributed_resource(
        self, resource: set[tuple[int, int]], count: int
    ) -> None:
        """Fill to ``count`` using incremental farthest-point placement."""
        if len(resource) >= count:
            return
        occupied = self.food | self.water | self.obstacles | {
            stockpile.position for stockpile in self.stockpiles.values()
        }
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        needed = count - len(resource)
        if needed > len(candidates):
            raise RuntimeError(
                f"World has room for only {len(resource) + len(candidates)} "
                f"distributed resources, not {count}"
            )
        if not resource:
            first = random.choice(candidates)
            resource.add(first)
            candidates.remove(first)
            needed -= 1
        nearest_distance = {
            position: min(
                self.manhattan_distance(position, existing)
                for existing in resource
            )
            for position in candidates
        }
        for _ in range(needed):
            maximum = max(nearest_distance.values())
            position = random.choice([
                candidate
                for candidate, distance in nearest_distance.items()
                if distance == maximum
            ])
            resource.add(position)
            del nearest_distance[position]
            for candidate, distance in nearest_distance.items():
                nearest_distance[candidate] = min(
                    distance, self.manhattan_distance(candidate, position)
                )

    def _spread_resource_seed(
        self, resource: set[tuple[int, int]]
    ) -> tuple[int, int]:
        """Place cluster centers far apart so the grid has no resource deserts."""
        occupied = self.food | self.water | self.obstacles
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if resource is self.water and self.food:
            habitat_candidates = [
                position
                for position in candidates
                if min(
                    self.manhattan_distance(position, food)
                    for food in self.food
                )
                <= 3
            ]
            if habitat_candidates:
                candidates = habitat_candidates
        if not candidates:
            raise RuntimeError("World has no free position for a resource cluster")
        if not resource:
            return random.choice(candidates)
        distances = {
            position: min(
                self.manhattan_distance(position, seed) for seed in resource
            )
            for position in candidates
        }
        maximum = max(distances.values())
        return random.choice([
            position for position, distance in distances.items() if distance == maximum
        ])

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
            and position not in {
                stockpile.position for stockpile in self.stockpiles.values()
            }
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
            needed = (
                new_position in self.food
                and agent.hunger >= self.config.priority_need_threshold
            ) or (
                new_position in self.water
                and agent.thirst >= self.config.priority_need_threshold
            )
            return ActionResult(action, reached_needed_resource=needed)
        if action is Action.EAT:
            position = self.resource_position_in_reach(agent, "food")
            stockpile = self.stockpile_for(agent)
            if position is not None:
                self.food.remove(position)
            elif agent.carried_food > 0:
                agent.carried_food -= 1
            elif (
                stockpile is not None
                and stockpile.food > 0
                and self.position_in_reach(agent, stockpile.position)
            ):
                stockpile.food -= 1
            else:
                return ActionResult(action, invalid=True)
            hunger_before = agent.hunger
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
        if action is Action.ATTACK:
            target = self.attack_target_in_reach(agent)
            if target is None:
                return ActionResult(action, invalid=True)
            health_before = target.health
            target.health = max(0.0, target.health - self.config.predator_attack_damage)
            damage = health_before - target.health
            target.last_health_delta = -damage
            killed = target.health <= 0.0
            if killed:
                target.alive = False
                target.cause_of_death = f"attack:{agent.id}"
            return ActionResult(
                action,
                attacked=True,
                attack_damage=damage,
                killed=killed,
                target_id=target.id,
            )
        if action is Action.GATHER:
            stockpile = self.stockpile_for(agent)
            hungry_baby = self.hungry_dependent_in_reach(agent)
            if agent.carried_food > 0 and hungry_baby is not None:
                agent.carried_food -= 1
                hungry_baby.hunger = max(
                    0.0, hungry_baby.hunger - self.config.food_hunger_reduction
                )
                return ActionResult(
                    action, fed_baby=True, target_id=hungry_baby.id
                )
            if (
                agent.carried_food > 0
                and stockpile is not None
                and self.position_in_reach(agent, stockpile.position)
            ):
                stockpile.food += agent.carried_food
                agent.carried_food = 0
                return ActionResult(action, deposited=True)
            position = self.resource_position_in_reach(agent, "food")
            if (
                position is None
                or agent.hunger >= self.config.gathering_hunger_limit
                or agent.carried_food >= self.config.gather_capacity
            ):
                return ActionResult(action, invalid=True)
            self.food.remove(position)
            hungry_baby = self.hungry_dependent_in_reach(agent)
            if hungry_baby is not None:
                hungry_baby.hunger = max(
                    0.0, hungry_baby.hunger - self.config.food_hunger_reduction
                )
                return ActionResult(
                    action, gathered=True, fed_baby=True, target_id=hungry_baby.id
                )
            agent.carried_food += 1
            return ActionResult(action, gathered=True)
        if action is Action.MATE:
            target = self.mate_target_in_reach(agent)
            if target is None:
                return ActionResult(action, invalid=True)
            female = agent if agent.sex == "F" else target
            male = target if female is agent else agent
            female.heart_partner_id = male.id
            male.heart_partner_id = female.id
            female.heart_ticks_remaining = self.config.courtship_ticks
            male.heart_ticks_remaining = self.config.courtship_ticks
            return ActionResult(action, mated=True, target_id=target.id)
        return ActionResult(action)

    def action_constraints(self, agent: BaseAgent) -> ActionConstraints:
        """Return valid actions, narrowed by the survival governor when urgent."""
        mask = [False for _ in Action]
        for action, (dx, dy) in MOVEMENT_DELTAS.items():
            destination = (agent.x + dx, agent.y + dy)
            mask[action] = self.in_bounds(*destination) and destination not in self.obstacles
        mask[Action.EAT] = self.food_available_in_reach(agent)
        mask[Action.DRINK] = self.resource_in_reach(agent, "water")
        mask[Action.REST] = True
        mask[Action.WAIT] = True
        mask[Action.ATTACK] = (
            self.attack_target_in_reach(agent) is not None
        )
        stockpile = self.stockpile_for(agent)
        can_deposit = (
            agent.carried_food > 0
            and stockpile is not None
            and self.position_in_reach(agent, stockpile.position)
        )
        can_feed_baby = (
            agent.carried_food > 0
            and self.hungry_dependent_in_reach(agent) is not None
        )
        can_collect = (
            agent.hunger < self.config.gathering_hunger_limit
            and agent.carried_food < self.config.gather_capacity
            and self.resource_in_reach(agent, "food")
        )
        mask[Action.GATHER] = can_feed_baby or can_deposit or can_collect
        mask[Action.MATE] = self.mate_target_in_reach(agent) is not None

        needs = {
            "food": agent.hunger,
            "water": agent.thirst,
            "rest": 1.0 - agent.energy,
        }
        memories = {
            "food": agent.spatial_memory.food.position,
            "water": agent.spatial_memory.water.position,
        }
        physical_movement_mask = mask.copy()
        for action, (dx, dy) in MOVEMENT_DELTAS.items():
            if not mask[action]:
                continue
            destination = (agent.x + dx, agent.y + dy)
            for name, need, rate in (
                ("food", agent.hunger, self.config.hunger_per_tick),
                ("water", agent.thirst, self.config.thirst_per_tick),
            ):
                target = memories[name]
                if target is None:
                    continue
                distance = self._distance_map(target).get(destination)
                travel_ticks = max(0, distance - 1) if distance is not None else 10_000
                available_ticks = (
                    (self.config.need_danger_threshold - need) / max(1e-9, rate)
                    - self.config.survival_travel_reserve_ticks
                )
                if travel_ticks > available_ticks:
                    mask[action] = False
                    break
        if not any(mask[action] for action in MOVEMENT_DELTAS):
            # If the current state is already outside the viability envelope,
            # movement toward the selected priority remains better than paralysis.
            for action in MOVEMENT_DELTAS:
                mask[action] = physical_movement_mask[action]
        travel_by_need: dict[str, int] = {"rest": 0}
        for name, need in needs.items():
            if name != "rest":
                target = memories[name]
                distance = (
                    self._distance_map(target).get((agent.x, agent.y))
                    if target is not None
                    else None
                )
                travel_by_need[name] = distance if distance is not None else 10_000
        urgency = max(needs.values())
        # Between planning and danger thresholds, the brain keeps ownership of
        # meal timing. The hard governor takes over only in the danger zone.
        if urgency < self.config.need_danger_threshold:
            return ActionConstraints(mask, "valid_actions")
        # When needs are nearly tied, finish the nearest one first. Distance
        # then decreases every tick, which naturally prevents route ping-pong.
        candidates = [
            name for name, need in needs.items() if urgency - need <= 0.02
        ]
        priority = min(candidates, key=lambda name: travel_by_need[name])

        if priority == "rest":
            return ActionConstraints(
                _only_actions(Action.REST), "survival", priority="rest"
            )

        consume_action = Action.EAT if priority == "food" else Action.DRINK
        if mask[consume_action]:
            return ActionConstraints(
                _only_actions(consume_action), "survival", priority=priority
            )

        memory = getattr(agent.spatial_memory, priority)
        target = memory.position
        if target is not None:
            route_actions = [
                action
                for action in self._shortest_route_actions((agent.x, agent.y), target)
                if mask[action]
            ]
            if route_actions:
                return ActionConstraints(
                    _only_actions(*route_actions),
                    "survival",
                    priority=priority,
                    target=target,
                )

        search_actions = [
            action for action in MOVEMENT_DELTAS if mask[action]
        ]
        if search_actions:
            return ActionConstraints(
                _only_actions(*search_actions), "search", priority=priority
            )
        # A fully enclosed agent must retain one valid action.
        return ActionConstraints(
            _only_actions(Action.WAIT), "trapped", priority=priority
        )

    def _shortest_route_actions(
        self, start: tuple[int, int], target: tuple[int, int]
    ) -> list[Action]:
        """Return every first move belonging to a shortest obstacle-safe path."""
        distances = self._distance_map(target)
        candidates: list[tuple[int, Action]] = []
        for action, (dx, dy) in MOVEMENT_DELTAS.items():
            destination = (start[0] + dx, start[1] + dy)
            if destination in distances:
                candidates.append((distances[destination], action))
        if not candidates:
            return []
        best = min(distance for distance, _ in candidates)
        return [action for distance, action in candidates if distance == best]

    def _distance_map(
        self, target: tuple[int, int]
    ) -> dict[tuple[int, int], int]:
        cached = self._distance_map_cache.get(target)
        if cached is not None:
            return cached
        distances = {target: 0}
        queue = deque([target])
        while queue:
            x, y = queue.popleft()
            for dx, dy in MOVEMENT_DELTAS.values():
                neighbour = (x + dx, y + dy)
                if (
                    self.in_bounds(*neighbour)
                    and neighbour not in self.obstacles
                    and neighbour not in distances
                ):
                    distances[neighbour] = distances[(x, y)] + 1
                    queue.append(neighbour)
        self._distance_map_cache[target] = distances
        return distances

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

    def stockpile_for(self, agent: BaseAgent) -> Stockpile | None:
        return self.stockpiles.get(agent.agent_type)

    def position_in_reach(
        self, agent: BaseAgent, position: tuple[int, int]
    ) -> bool:
        return self.manhattan_distance((agent.x, agent.y), position) <= 1

    def food_available_in_reach(self, agent: BaseAgent) -> bool:
        stockpile = self.stockpile_for(agent)
        return (
            self.resource_in_reach(agent, "food")
            or agent.carried_food > 0
            or (
                stockpile is not None
                and stockpile.food > 0
                and self.position_in_reach(agent, stockpile.position)
            )
        )

    def mate_target_in_reach(self, agent: BaseAgent) -> BaseAgent | None:
        """Return a same-species F/M partner sharing the exact grid cell."""
        if not agent.alive or agent.dependent_ticks_remaining > 0:
            return None
        if agent.heart_ticks_remaining > 0:
            return None
        if agent.sex == "F" and (
            agent.pregnancy_ticks_remaining > 0 or agent.dependent_ids
        ):
            return None
        candidates = [
            other
            for other in self.living_agents
            if other.id != agent.id
            and other.agent_type == agent.agent_type
            and other.sex != agent.sex
            and (other.x, other.y) == (agent.x, agent.y)
            and other.dependent_ticks_remaining <= 0
            and other.heart_ticks_remaining <= 0
            and not (
                other.sex == "F"
                and (other.pregnancy_ticks_remaining > 0 or other.dependent_ids)
            )
        ]
        return min(candidates, key=lambda other: other.id, default=None)

    def hungry_dependent_in_reach(self, mother: BaseAgent) -> BaseAgent | None:
        dependents = [
            agent
            for agent in self.living_agents
            if agent.id in mother.dependent_ids
            and agent.dependent_ticks_remaining > 0
            and agent.hunger >= self.config.need_action_threshold
            and self.position_in_reach(mother, (agent.x, agent.y))
        ]
        return max(dependents, key=lambda agent: agent.hunger, default=None)

    def max_dependent_hunger(self, mother: BaseAgent) -> float:
        dependents = [
            agent.hunger
            for agent in self.living_agents
            if agent.id in mother.dependent_ids
            and agent.dependent_ticks_remaining > 0
        ]
        return max(dependents, default=0.0)

    def attack_target_in_reach(self, agent: BaseAgent) -> BaseAgent | None:
        """Return the nearest legal target for a predator or defending human."""
        if agent.agent_type == "human":
            legal_target = lambda other: (
                other.agent_type == "animal" and other.predator
            )
        elif agent.agent_type == "animal" and agent.predator:
            legal_target = lambda other: True
        else:
            return None
        candidates = [
            other
            for other in self.living_agents
            if other.id != agent.id
            and legal_target(other)
            and self.manhattan_distance((agent.x, agent.y), (other.x, other.y)) <= 1
        ]
        return min(candidates, key=lambda other: other.id, default=None)

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


def _only_actions(*actions: Action) -> list[bool]:
    allowed = set(actions)
    return [action in allowed for action in Action]

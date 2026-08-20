from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

import torch
import pytest

from agents.animal import Animal
from agents.brain import AgentBrain
from agents.human import Human
from config import BrainConfig, SimulationConfig
from learning.replay_buffer import Experience, ReplayBuffer
from learning.rewards import (
    calculate_need_safety_signal,
    calculate_reward,
    calculate_survival_priority_penalty,
)
from persistence.checkpoints import (
    CheckpointError,
    _load_widened_state_dict,
    load_agents,
    model_hash,
    read_metadata,
    save_run_checkpoints,
)
from persistence.best_result import _score, brb_public_summary, load_brb_payloads
from simulation.engine import SimulationEngine
from simulation.experiment import build_brb_engine, build_resumed_engine, run_continuous
from web.server import WebSimulationController
from world.grid import Action
from world.world import World


def small_config() -> SimulationConfig:
    return SimulationConfig(
        grid_width=8,
        grid_height=6,
        num_humans=1,
        num_animals=1,
        food_per_agent=1,
        initial_food=2,
        initial_water=2,
        initial_obstacles=1,
        food_cluster_count=1,
        water_cluster_count=1,
        max_food=10,
        human_brain=BrainConfig(hidden_sizes=[4, 8, 8], batch_size=2, train_every=1),
        animal_brain=BrainConfig(hidden_sizes=[4, 6, 6], batch_size=2, train_every=1),
    )


def test_world_can_be_created() -> None:
    world = World(small_config())
    assert (world.width, world.height) == (8, 6)


def test_agents_can_be_created_with_independent_brains() -> None:
    world = World(small_config())
    assert len(world.agents) == 2
    assert isinstance(world.agents[0], Human)
    assert isinstance(world.agents[1], Animal)
    assert world.agents[0].brain is not world.agents[1].brain


def test_agents_have_balanced_binary_sex_and_animals_have_predator_flags() -> None:
    config = small_config()
    config.num_humans = 4
    config.num_animals = 4
    config.predator_fraction = 0.5
    world = World(config)
    humans = [agent for agent in world.agents if agent.agent_type == "human"]
    animals = [agent for agent in world.agents if agent.agent_type == "animal"]
    assert [agent.sex for agent in humans] == ["F", "M", "F", "M"]
    assert [agent.predator for agent in animals] == [True, True, False, False]
    assert not any(agent.predator for agent in humans)


def test_food_supply_scales_with_inhabitants_and_is_distributed() -> None:
    config = small_config()
    config.num_humans = 3
    config.num_animals = 2
    config.initial_food = 1
    config.food_per_agent = 3
    world = World(config)
    assert world.food_target == 15
    assert len(world.food) == 15
    assert len({x for x, _ in world.food}) > 1
    assert len({y for _, y in world.food}) > 1


def test_predators_can_attack_adjacent_agents() -> None:
    config = small_config()
    config.predator_attack_damage = 0.5
    world = World(config, populate=False)
    predator = Animal.create("animal_001", 2, 2, config)
    prey = Human.create("human_001", 3, 2, config)
    non_predator = Animal.create("animal_002", 4, 3, config)
    non_predator.predator = False
    world.agents = [predator, prey, non_predator]

    assert world.action_constraints(predator).mask[Action.ATTACK]
    first = world.execute_action(predator, Action.ATTACK)
    second = world.execute_action(predator, Action.ATTACK)
    assert first.attacked and first.target_id == prey.id
    assert second.killed and not prey.alive
    assert prey.cause_of_death == "attack:animal_001"


def test_humans_can_attack_predators_but_not_prey_or_other_humans() -> None:
    config = small_config()
    config.predator_attack_damage = 0.5
    world = World(config, populate=False)
    human = Human.create("human_001", 2, 2, config)
    predator = Animal.create("animal_001", 3, 2, config)
    prey = Animal.create("animal_002", 2, 3, config)
    prey.predator = False
    neighbour = Human.create("human_002", 1, 2, config)
    world.agents = [human, predator, prey, neighbour]

    assert world.action_constraints(human).mask[Action.ATTACK]
    first = world.execute_action(human, Action.ATTACK)
    second = world.execute_action(human, Action.ATTACK)
    assert first.target_id == predator.id
    assert second.killed
    assert predator.cause_of_death == "attack:human_001"
    assert not world.action_constraints(human).mask[Action.ATTACK]
    assert world.execute_action(human, Action.ATTACK).invalid


def test_gathering_moves_food_into_a_communal_stockpile() -> None:
    config = small_config()
    world = World(config, populate=False)
    human = Human.create("human_001", 2, 2, config)
    world.agents = [human]
    world.initialize_stockpiles()
    world.food.add((3, 2))

    assert world.action_constraints(human).mask[Action.GATHER]
    gathered = world.execute_action(human, Action.GATHER)
    assert gathered.gathered
    assert human.carried_food == 1
    assert not world.food
    assert world.total_food_supply == 1

    deposited = world.execute_action(human, Action.GATHER)
    assert deposited.deposited
    assert human.carried_food == 0
    assert world.stockpiles["human"].food == 1
    assert world.total_food_supply == 1


def test_gathering_is_available_only_while_hunger_is_below_limit() -> None:
    config = small_config()
    world = World(config, populate=False)
    human = Human.create("human_001", 2, 2, config)
    world.agents = [human]
    world.initialize_stockpiles()
    world.food.add((3, 2))
    human.hunger = config.gathering_hunger_limit
    assert not world.action_constraints(human).mask[Action.GATHER]


def test_agents_gather_wood_and_build_one_shared_house() -> None:
    config = small_config()
    config.house_materials_required = 2
    world = World(config, populate=False)
    first = Human.create("human_001", 2, 2, config)
    second = Human.create("human_002", 2, 2, config)
    world.agents = [first, second]
    world.initialize_stockpiles()
    house = world.houses["human"]

    world.trees[(3, 2)] = 0
    assert world.action_constraints(first).mask[Action.GATHER]
    gathered = world.execute_action(first, Action.GATHER)
    assert gathered.gathered and gathered.gathered_wood
    assert first.carried_wood == 1
    assert (3, 2) not in world.trees
    assert world.action_constraints(first).mask[Action.BUILD]
    built = world.execute_action(first, Action.BUILD)
    assert built.built and not built.house_completed
    assert house.materials == 1

    world.trees[(2, 3)] = 1
    world.execute_action(second, Action.GATHER)
    completed = world.execute_action(second, Action.BUILD)
    assert completed.built and completed.house_completed
    assert world.house_for(first) is world.house_for(second) is house
    assert house.complete
    assert world.agent_has_shelter(first)
    assert not world.action_constraints(first).mask[Action.BUILD]


def test_group_house_improves_rest_for_every_member() -> None:
    config = small_config()
    config.house_materials_required = 1
    world = World(config, populate=False)
    builder = Human.create("human_001", 2, 2, config)
    neighbour = Human.create("human_002", 2, 2, config)
    world.agents = [builder, neighbour]
    world.initialize_stockpiles()
    builder.carried_wood = 1
    assert world.execute_action(builder, Action.BUILD).house_completed

    neighbour.energy = 0.20
    neighbour.update_biology(
        shelter_energy_gain=config.house_passive_energy_gain,
    )
    assert neighbour.energy == pytest.approx(
        0.20 - config.energy_per_tick + config.house_passive_energy_gain
    )

    neighbour.energy = 0.20
    result = world.execute_action(neighbour, Action.REST)
    assert result.sheltered
    neighbour.update_biology(
        rested=True,
        rest_multiplier=config.house_rest_multiplier,
        shelter_energy_gain=config.house_passive_energy_gain,
    )
    assert neighbour.energy == pytest.approx(
        0.20
        + config.rest_energy_gain * config.house_rest_multiplier
        + config.house_passive_energy_gain
    )


def test_group_house_protects_occupants_from_predators() -> None:
    config = small_config()
    config.house_materials_required = 1
    world = World(config, populate=False)
    predator = Animal.create("animal_001", 2, 2, config)
    protected = Human.create("human_001", 3, 2, config)
    world.agents = [predator, protected]
    world.initialize_stockpiles()
    world.houses["human"].materials = 1

    assert world.agent_has_shelter(protected)
    assert world.attack_target_in_reach(predator) is None
    assert not world.action_constraints(predator).mask[Action.ATTACK]
    assert world.execute_action(predator, Action.ATTACK).invalid

    outsider = Human.create("human_002", 1, 2, config)
    world.agents.append(outsider)
    assert not world.agent_has_shelter(outsider)
    assert world.attack_target_in_reach(predator) is outsider


def test_construction_rewards_are_explicit_and_inspectable() -> None:
    reward = calculate_reward(
        ate=False,
        drank=False,
        invalid=False,
        reached_needed_resource=False,
        health_delta=0.0,
        died=False,
        repeating=False,
        wood_gather_reward=0.20,
        house_build_reward=0.35,
        house_completion_reward=1.50,
        construction_progress=0.08,
        sheltered_rest_reward=0.30,
    )
    assert reward.components["gathered_wood"] == 0.20
    assert reward.components["built_group_house"] == 0.35
    assert reward.components["completed_group_house"] == 1.50
    assert reward.components["construction_progress"] == 0.08
    assert reward.components["rested_in_group_house"] == 0.30


def test_gathering_progress_has_dense_directional_reward() -> None:
    closer = calculate_reward(
        ate=False, drank=False, invalid=False, reached_needed_resource=False,
        health_delta=0.0, died=False, repeating=False,
        gathering_progress=0.08,
    )
    farther = calculate_reward(
        ate=False, drank=False, invalid=False, reached_needed_resource=False,
        health_delta=0.0, died=False, repeating=False,
        gathering_progress=-0.04,
    )
    assert closer.components["gathering_progress"] == 0.08
    assert farther.components["gathering_progress"] == -0.04


@pytest.mark.parametrize(
    ("roll", "expected_babies"), [(0.10, 1), (0.80, 2), (0.98, 3)]
)
def test_birth_litter_probabilities_map_to_one_two_or_three_babies(
    monkeypatch: pytest.MonkeyPatch, roll: float, expected_babies: int
) -> None:
    config = small_config()
    world = World(config, populate=False)
    mother = Human.create("human_001", 2, 2, config)
    world.agents = [mother]
    monkeypatch.setattr("world.world.random.random", lambda: roll)
    babies = world._give_birth(mother)
    assert len(babies) == expected_babies
    assert mother.children_born == expected_babies
    assert all(baby.mother_id == mother.id for baby in babies)
    assert all(
        baby.dependent_ticks_remaining == config.dependent_baby_ticks
        for baby in babies
    )


def test_mating_respects_full_heart_and_pregnancy_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_config()
    world = World(config, populate=False)
    female = Human.create("human_001", 2, 2, config)
    male = Human.create("human_002", 2, 2, config)
    world.agents = [female, male]
    world.initialize_stockpiles()
    monkeypatch.setattr("world.world.random.random", lambda: 0.10)

    result = world.execute_action(female, Action.MATE)
    assert result.mated
    assert female.heart_ticks_remaining == male.heart_ticks_remaining == 50
    for remaining in range(config.courtship_ticks - 1, 0, -1):
        assert not world.advance_social_dynamics()
        assert female.heart_ticks_remaining == remaining
    assert not world.advance_social_dynamics()
    assert female.pregnancy_ticks_remaining == 300
    assert not world.action_constraints(female).mask[Action.MATE]

    for remaining in range(config.pregnancy_ticks - 1, 0, -1):
        assert not world.advance_social_dynamics()
        assert female.pregnancy_ticks_remaining == remaining
    babies = world.advance_social_dynamics()
    assert len(babies) == 1
    assert female.pregnancy_ticks_remaining == 0
    assert female.dependent_ids == [babies[0].id]
    assert not world.action_constraints(female).mask[Action.MATE]


def test_dependent_baby_follows_mother_and_receives_carried_food(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_config()
    world = World(config, populate=False)
    mother = Human.create("human_001", 4, 2, config)
    world.agents = [mother]
    monkeypatch.setattr("world.world.random.random", lambda: 0.10)
    baby = world._give_birth(mother)[0]
    baby.x = 2
    mother.carried_food = 1
    newborns = world.advance_social_dynamics()
    assert not newborns
    assert (baby.x, baby.y) == (3, 2)
    assert baby.dependent_ticks_remaining == config.dependent_baby_ticks - 1
    fed = world.execute_action(mother, Action.GATHER)
    assert fed.fed_baby
    assert baby.hunger == pytest.approx(0.0)
    assert mother.carried_food == 0


def test_newborn_brain_joins_species_horde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = small_config()
    config.num_ticks = 1
    config.stop_when_no_humans = False
    world = World(config, populate=False)
    mother = Human.create("human_001", 2, 2, config)
    mother.pregnancy_ticks_remaining = 1
    world.agents = [mother]
    world.initialize_stockpiles()
    engine = SimulationEngine(world, config, 1, 1, 42, tmp_path)
    monkeypatch.setattr("world.world.random.random", lambda: 0.10)
    assert engine.step()
    baby = next(agent for agent in world.agents if agent.id != mother.id)
    assert baby.dependent_ticks_remaining == config.dependent_baby_ticks
    assert baby.trainer.horde_replay_buffer is engine.horde_replay_buffers["human"]


def test_mother_is_penalized_for_hungry_and_starved_dependent_baby(
    tmp_path: Path,
) -> None:
    config = small_config()
    config.num_ticks = 3
    config.stop_when_no_humans = False
    world = World(config, populate=False)
    mother = Human.create("human_001", 2, 2, config)
    baby = Human.create("human_003", 2, 2, config)
    baby.mother_id = mother.id
    baby.dependent_ticks_remaining = config.dependent_baby_ticks
    baby.hunger = 1.0
    baby.health = 0.01
    mother.dependent_ids = [baby.id]
    mother.children_born = 1
    world.agents = [mother, baby]
    world.initialize_stockpiles()
    engine = SimulationEngine(world, config, 1, 1, 42, tmp_path)

    assert engine.step()
    first_components = engine.last_agent_events[mother.id]["reward_components"]
    assert first_components["hungry_dependent_baby"] < 0.0
    assert not baby.alive
    assert engine.pending_maternal_starvation_penalties[mother.id] == 5.0

    assert engine.step()
    second_components = engine.last_agent_events[mother.id]["reward_components"]
    assert second_components["dependent_baby_starved"] == -5.0


def test_next_cycle_accepts_population_expanded_by_births(tmp_path: Path) -> None:
    config = small_config()
    config.compact_console = True
    config.generate_plots = False
    world = World(config)
    baby = Human.create("human_002", 4, 4, config)
    world.agents.append(baby)
    checkpoint_dir = tmp_path / "checkpoints" / "experiment_001" / "run_001"
    save_run_checkpoints(world.agents, checkpoint_dir, 1, 1, 42, config, None, {})
    resumed = build_resumed_engine(tmp_path, checkpoint_dir, config, 43)
    assert len(resumed.world.agents) == 3
    assert {agent.id for agent in resumed.world.agents} == {
        "human_001", "human_002", "animal_001"
    }


def test_individual_epsilon_profiles_have_a_small_scout_minority() -> None:
    config = small_config()
    config.num_humans = 15
    world = World(config)
    humans = [agent for agent in world.agents if agent.agent_type == "human"]
    scouts = [agent for agent in humans if agent.exploration_profile == "scout"]
    standard = [agent for agent in humans if agent.exploration_profile == "standard"]
    assert len(scouts) == 2
    assert all(agent.exploration_epsilon == 0.50 for agent in scouts)
    assert all(0.01 <= agent.exploration_epsilon <= 0.15 for agent in standard)


def test_horde_replay_is_shared_by_species_but_brains_remain_independent(
    tmp_path: Path,
) -> None:
    config = small_config()
    config.num_humans = 2
    config.num_animals = 2
    engine = SimulationEngine(World(config), config, 1, 1, 42, tmp_path)
    humans = [agent for agent in engine.world.agents if agent.agent_type == "human"]
    animals = [agent for agent in engine.world.agents if agent.agent_type == "animal"]
    assert humans[0].trainer.horde_replay_buffer is humans[1].trainer.horde_replay_buffer
    assert animals[0].trainer.horde_replay_buffer is animals[1].trainer.horde_replay_buffer
    assert humans[0].trainer.horde_replay_buffer is not animals[0].trainer.horde_replay_buffer
    assert humans[0].brain is not humans[1].brain
    engine.step()
    assert len(humans[0].trainer.replay_buffer) == 1
    assert len(humans[0].trainer.learning_replay_buffer) == 2
    assert len(animals[0].trainer.learning_replay_buffer) == 2
    assert all(agent.trainer.training_steps > 0 for agent in humans)
    engine.step()
    assert len(humans[0].trainer.learning_replay_buffer) == 4


def test_brain_output_shape() -> None:
    brain = AgentBrain(
        input_size=12, hidden_sizes=[8, 16, 12], output_size=8, need_input_size=4
    )
    assert brain(torch.zeros(4, 12)).shape == (4, 8)


def test_brain_v2_exposes_branch_activations() -> None:
    brain = AgentBrain(
        input_size=12, hidden_sizes=[8, 16, 12], output_size=8, need_input_size=4
    )
    output, activations = brain.forward_with_activations(torch.zeros(2, 12))
    assert output.shape == (2, 8)
    assert activations["need_inputs"].shape == (2, 4)
    assert activations["spatial_inputs"].shape == (2, 8)
    assert activations["need_hidden"].shape == (2, 8)
    assert activations["spatial_hidden"].shape == (2, 16)
    assert activations["fusion_hidden"].shape == (2, 12)
    assert brain.architecture["architecture_version"] == 2


def test_agent_can_move() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    result = world.execute_action(agent, Action.MOVE_RIGHT)
    assert not result.invalid
    assert (agent.x, agent.y) == (3, 2)


def test_water_is_impassable_for_movement_and_action_mask() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.water.add((3, 2))

    constraints = world.action_constraints(agent)
    result = world.execute_action(agent, Action.MOVE_RIGHT)

    assert not constraints.mask[Action.MOVE_RIGHT]
    assert constraints.mask[Action.DRINK]
    assert result.invalid
    assert (agent.x, agent.y) == (2, 2)
    assert world.cardinal_obstacle_flags(2, 2) == [0.0, 0.0, 0.0, 1.0]


def test_shortest_route_goes_around_water() -> None:
    config = small_config()
    world = World(config, populate=False)
    world.water.add((2, 1))

    actions = world._shortest_route_actions((1, 1), (3, 1))

    assert Action.MOVE_RIGHT not in actions
    assert set(actions) == {Action.MOVE_UP, Action.MOVE_DOWN}


def test_action_mask_removes_physically_impossible_actions() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 0, 0, config)
    world.agents = [agent]
    constraints = world.action_constraints(agent)
    assert not constraints.mask[Action.MOVE_UP]
    assert not constraints.mask[Action.MOVE_LEFT]
    assert not constraints.mask[Action.EAT]
    assert not constraints.mask[Action.DRINK]
    assert constraints.mask[Action.MOVE_RIGHT]


def test_survival_governor_forces_consumption_when_urgent_resource_is_reachable() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.water.add((3, 2))
    agent.thirst = 0.75
    constraints = world.action_constraints(agent)
    assert constraints.mode == "survival"
    assert constraints.priority == "water"
    assert [Action(index) for index, allowed in enumerate(constraints.mask) if allowed] == [
        Action.DRINK
    ]


def test_survival_governor_routes_toward_remembered_resource() -> None:
    config = small_config()
    config.resource_sense_radius = 8
    world = World(config, populate=False)
    agent = Human.create("human_001", 1, 1, config)
    world.agents = [agent]
    world.food.add((5, 1))
    agent.hunger = 0.75
    agent.perceive(world)
    constraints = world.action_constraints(agent)
    assert constraints.target == (5, 1)
    assert constraints.mask[Action.MOVE_RIGHT]
    assert sum(constraints.mask) == 1


def test_brain_controls_meal_timing_before_hunger_becomes_dangerous() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.food.add((3, 2))
    agent.hunger = 0.50
    constraints = world.action_constraints(agent)
    assert constraints.mode == "valid_actions"
    assert constraints.mask[Action.EAT]
    assert constraints.mask[Action.WAIT]


def test_brain_action_selection_respects_governor_mask() -> None:
    config = small_config()
    agent = Human.create("human_001", 1, 1, config)
    state = torch.zeros(Human.INPUT_SIZE)
    mask = [action is Action.MOVE_RIGHT for action in Action]
    action = agent.choose_action(state, action_mask=mask)
    assert action == Action.MOVE_RIGHT
    assert agent.last_allowed_actions == ["MOVE_RIGHT"]


def test_eating_modifies_hunger() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.hunger = 0.8
    world.food.add((2, 2))
    assert world.execute_action(agent, Action.EAT).ate
    assert agent.hunger < 0.8


def test_drinking_modifies_thirst() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.thirst = 0.8
    world.water.add((2, 2))
    water_before = world.water.copy()
    assert world.execute_action(agent, Action.DRINK).drank
    assert agent.thirst < 0.8
    assert world.water == water_before


def test_agent_can_drink_from_adjacent_water() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.thirst = 0.8
    world.water.add((3, 2))
    result = world.execute_action(agent, Action.DRINK)
    assert result.drank
    assert agent.thirst < 0.8
    assert (3, 2) in world.water


def test_unnecessary_drinking_has_escalating_penalty() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.water.add((2, 2))
    agent.thirst = 0.01

    first = world.execute_action(agent, Action.DRINK)
    first_penalty = agent.unnecessary_need_penalty(
        first.action.name, first.unnecessary_need_action
    )
    agent.thirst = 0.01
    second = world.execute_action(agent, Action.DRINK)
    second_penalty = agent.unnecessary_need_penalty(
        second.action.name, second.unnecessary_need_action
    )

    assert not first.drank and not second.drank
    assert first_penalty == 0.1
    assert second_penalty == 0.2
    reward = calculate_reward(
        ate=False, drank=False, rested=False, invalid=False,
        reached_needed_resource=False, health_delta=0.0, died=False,
        repeating=False, unnecessary_action_penalty=second_penalty,
    )
    assert reward.components["unnecessary_need_action"] == -0.2


def test_wait_is_penalized_more_when_survival_is_critical() -> None:
    healthy = calculate_survival_priority_penalty(
        action_name="WAIT", hunger=0.20, thirst=0.20, energy=0.90,
        health=1.0, threshold=0.50, base=0.05, scale=0.25, cap=0.60,
    )
    urgent = calculate_survival_priority_penalty(
        action_name="WAIT", hunger=1.0, thirst=0.70, energy=0.50,
        health=0.20, threshold=0.50, base=0.05, scale=0.25, cap=0.60,
    )
    correct = calculate_survival_priority_penalty(
        action_name="EAT", hunger=1.0, thirst=0.70, energy=0.50,
        health=0.20, threshold=0.50, base=0.05, scale=0.25, cap=0.60,
    )
    searching = calculate_survival_priority_penalty(
        action_name="MOVE_LEFT", hunger=1.0, thirst=0.70, energy=0.50,
        health=0.20, threshold=0.50, base=0.05, scale=0.25, cap=0.60,
    )
    assert healthy == 0.0
    assert urgent == pytest.approx(0.54)
    assert correct == 0.0
    assert searching == 0.0


def test_wrong_movement_is_penalized_when_an_urgent_target_is_known() -> None:
    closer = calculate_survival_priority_penalty(
        action_name="MOVE_LEFT", hunger=0.60, thirst=0.20, energy=0.90,
        health=1.0, threshold=0.35, base=0.05, scale=0.25, cap=0.60,
        movement_has_known_target=True, resource_progress=0.11,
    )
    wandering = calculate_survival_priority_penalty(
        action_name="MOVE_RIGHT", hunger=0.60, thirst=0.20, energy=0.90,
        health=1.0, threshold=0.35, base=0.05, scale=0.25, cap=0.60,
        movement_has_known_target=True, resource_progress=-0.10,
    )
    searching = calculate_survival_priority_penalty(
        action_name="MOVE_RIGHT", hunger=0.60, thirst=0.20, energy=0.90,
        health=1.0, threshold=0.35, base=0.05, scale=0.25, cap=0.60,
        movement_has_known_target=False, resource_progress=0.0,
    )
    assert closer == 0.0
    assert wandering > 0.0
    assert searching == 0.0


def test_need_safety_signal_grows_steeply_between_fifty_and_seventy_percent() -> None:
    settings = dict(
        hunger_before=0.49, thirst_before=0.20,
        safe_target=0.50, danger_threshold=0.70,
        penalty_at_danger=0.30, penalty_cap=0.80, recovery_reward=0.25,
    )
    safe = calculate_need_safety_signal(
        **settings, hunger_after=0.50, thirst_after=0.21
    )
    warning = calculate_need_safety_signal(
        **settings, hunger_after=0.60, thirst_after=0.21
    )
    danger = calculate_need_safety_signal(
        **settings, hunger_after=0.70, thirst_after=0.21
    )
    assert safe == 0.0
    assert warning == pytest.approx(-0.075)
    assert danger == pytest.approx(-0.30)


def test_returning_below_fifty_percent_earns_a_safety_reward() -> None:
    signal = calculate_need_safety_signal(
        hunger_before=0.65, thirst_before=0.30,
        hunger_after=0.16, thirst_after=0.31,
        safe_target=0.50, danger_threshold=0.70,
        penalty_at_danger=0.30, penalty_cap=0.80, recovery_reward=0.25,
    )
    assert signal == pytest.approx(0.25)


def test_survival_priority_penalty_is_visible_in_reward_components() -> None:
    reward = calculate_reward(
        ate=False, drank=False, rested=False, invalid=False,
        reached_needed_resource=False, health_delta=0.0, died=False,
        repeating=False, survival_priority_penalty=0.30,
    )
    assert reward.components["ignored_survival_priority"] == -0.30
    assert reward.total == pytest.approx(-0.29)


def test_need_reward_scales_with_thirst_and_resets_penalty() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.water.add((2, 2))
    agent.unnecessary_need_streaks["DRINK"] = 4
    agent.thirst = 0.8

    result = world.execute_action(agent, Action.DRINK)
    penalty = agent.unnecessary_need_penalty(
        result.action.name, result.unnecessary_need_action
    )
    reward = calculate_reward(
        ate=False, drank=result.drank, rested=False, invalid=False,
        reached_needed_resource=False, health_delta=0.0, died=False,
        repeating=False, need_reward=result.necessity * config.need_reward_scale,
    )

    assert result.drank
    assert penalty == 0.0
    assert agent.unnecessary_need_streaks["DRINK"] == 0
    assert math.isclose(reward.components["drank_while_thirsty"], 0.8)


def test_resources_are_populated_in_clusters() -> None:
    world = World(small_config(), populate=False)
    world._populate_resource_clusters(world.food, count=8, cluster_count=1)
    assert len(world.food) == 8
    assert any(
        abs(x1 - x2) + abs(y1 - y2) == 1
        for x1, y1 in world.food
        for x2, y2 in world.food
        if (x1, y1) != (x2, y2)
    )
    world.food.clear()
    world._populate_resource_clusters(world.water, count=8, cluster_count=1)
    assert len(world.water) == 8
    assert any(
        abs(x1 - x2) + abs(y1 - y2) == 1
        for x1, y1 in world.water
        for x2, y2 in world.water
        if (x1, y1) != (x2, y2)
    )


def test_food_reserve_is_replenished_in_distributed_positions() -> None:
    config = small_config()
    config.food_respawn_probability = 1.0
    world = World(config, populate=False)
    world.food.add((3, 3))
    world.update_resources()
    assert len(world.food) == 2
    assert any(abs(x - 3) + abs(y - 3) > 1 for x, y in world.food)


def test_consumed_food_remains_scarce_until_respawn_succeeds() -> None:
    config = small_config()
    config.food_respawn_probability = 0.0
    world = World(config, populate=False)
    world.food.add((3, 3))
    world.update_resources()
    assert len(world.food) == 1


def test_default_world_uses_dense_matrix() -> None:
    config = SimulationConfig()
    assert (config.grid_width, config.grid_height) == (60, 40)


def test_visual_biomes_form_regions_and_trees_form_forests() -> None:
    random_state = random.getstate()
    random.seed(42)
    try:
        world = World(SimulationConfig(), populate=False)
        world._populate_decorative_trees()
    finally:
        random.setstate(random_state)

    terrain_kinds = {cell for row in world.terrain for cell in row}
    matching_neighbours = 0
    neighbour_pairs = 0
    for y, row in enumerate(world.terrain):
        for x, terrain in enumerate(row):
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if nx >= world.width or ny >= world.height:
                    continue
                neighbour_pairs += 1
                matching_neighbours += world.terrain[ny][nx] == terrain

    assert terrain_kinds == {"grassland", "earth", "forest"}
    assert matching_neighbours / neighbour_pairs > 0.85
    assert len(world.trees) > 20
    assert set(world.trees.values()) == {0, 1, 2}
    assert all(world.terrain[y][x] == "forest" for x, y in world.trees)


def test_resource_progress_adds_dense_reward() -> None:
    closer = calculate_reward(
        ate=False, drank=False, invalid=False, reached_needed_resource=False,
        health_delta=0.0, died=False, repeating=False, resource_progress=0.05,
    )
    farther = calculate_reward(
        ate=False, drank=False, invalid=False, reached_needed_resource=False,
        health_delta=0.0, died=False, repeating=False, resource_progress=-0.03,
    )
    assert closer.components["resource_progress"] == 0.05
    assert closer.total > farther.total


def test_perception_reports_when_water_is_in_reach() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    world.water.add((3, 2))
    perception = agent.perceive(world)
    assert perception.shape == (Human.INPUT_SIZE,)
    food_reach = Human.NEED_INPUT_SIZE + Human.SPATIAL_LABELS.index("comida al alcance")
    water_reach = Human.NEED_INPUT_SIZE + Human.SPATIAL_LABELS.index("agua al alcance")
    assert perception[food_reach].item() == 0.0
    assert perception[water_reach].item() == 1.0


def test_brain_v2_starts_planning_before_fifty_percent() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.hunger = 0.50
    agent.thirst = 0.24
    agent.energy = 0.50
    agent.health = 0.49
    perception = agent.perceive(world)
    assert perception[4:8].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_survival_perception_exposes_damage_cause_and_time_margin() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.health = 0.25
    agent.hunger = 1.0
    agent.thirst = 0.80
    agent.energy = 0.20
    agent.last_health_delta = -config.starvation_damage
    perception = agent.perceive(world)
    values = {
        label: perception[index].item()
        for index, label in enumerate(Human.NEED_LABELS)
    }
    assert values["riesgo hambre"] == 1.0
    assert values["riesgo sed"] == pytest.approx((0.80 - 0.25) / 0.75)
    assert values["recibiendo daño"] == 1.0
    assert values["daño reciente"] > 0.0
    assert values["margen de vida"] == pytest.approx(0.10)
    assert values["urgencia vital"] == 1.0


def test_spatial_memory_keeps_last_seen_water_outside_local_vision() -> None:
    config = small_config()
    config.vision_radius = 1
    config.resource_sense_radius = 1
    config.spatial_memory_max_age = 10
    world = World(config, populate=False)
    agent = Human.create("human_001", 1, 1, config)
    world.agents = [agent]
    world.water = {(5, 1)}
    agent.perceive(world)
    assert agent.spatial_memory.water.position is None

    agent.x = 4
    agent.perceive(world)
    assert agent.spatial_memory.water.position == (5, 1)
    agent.x = 1
    agent.age = 5
    perception = agent.perceive(world)
    confidence_index = Human.NEED_INPUT_SIZE + Human.SPATIAL_LABELS.index("conf agua")
    assert agent.spatial_memory.water.position == (5, 1)
    assert perception[confidence_index].item() == 0.5

    world.water.clear()
    agent.x = 4
    agent.perceive(world)
    assert agent.spatial_memory.water.position is None


def test_experience_enters_replay_buffer() -> None:
    buffer = ReplayBuffer(3)
    experience = Experience(torch.zeros(2), 0, 1.0, torch.ones(2), False)
    buffer.push(experience)
    assert len(buffer) == 1
    assert buffer.sample(1)[0].reward == 1.0


def test_replay_preserves_next_action_mask() -> None:
    buffer = ReplayBuffer(3)
    mask = torch.tensor([True, False, True, False, True, False, True, False])
    buffer.push(
        Experience(torch.zeros(2), 0, 1.0, torch.ones(2), False, mask)
    )
    restored = ReplayBuffer(3)
    restored.load_state_dict(buffer.state_dict(), input_size=2)
    assert torch.equal(restored.sample(1)[0].next_action_mask, mask)


def test_dqn_target_ignores_actions_disabled_by_governor() -> None:
    config = small_config()
    agent = Human.create("human_001", 1, 1, config)
    with torch.no_grad():
        for parameter in agent.brain.parameters():
            parameter.zero_()
        for parameter in agent.trainer.target_brain.parameters():
            parameter.zero_()
        agent.trainer.target_brain.action_head.bias[Action.WAIT] = 100.0
        agent.trainer.target_brain.action_head.bias[Action.MOVE_UP] = 1.0
    mask = torch.tensor([action is Action.MOVE_UP for action in Action])
    state = torch.zeros(Human.INPUT_SIZE)
    for _ in range(config.human_brain.batch_size):
        agent.trainer.remember(
            state, Action.MOVE_UP, 0.0, state.clone(), False, mask
        )
    loss = agent.trainer.train_step(force=True)
    assert loss is not None
    assert loss < 1.0


def test_replay_migration_inserts_new_needs_before_spatial_inputs() -> None:
    old_state = torch.arange(26, dtype=torch.float32)
    buffer = ReplayBuffer(3)
    buffer.load_state_dict(
        [{
            "state": old_state,
            "action": 0,
            "reward": 1.0,
            "next_state": old_state + 1,
            "done": False,
        }],
        input_size=33,
        source_need_input_size=8,
        target_need_input_size=15,
    )
    migrated = buffer.sample(1)[0].state
    assert torch.equal(migrated[:8], old_state[:8])
    assert torch.equal(migrated[8:15], torch.zeros(7))
    assert torch.equal(migrated[15:], old_state[8:])


def test_replay_sampling_reuses_rare_positive_experience() -> None:
    buffer = ReplayBuffer(20)
    for index in range(10):
        buffer.push(
            Experience(
                torch.tensor([float(index)]), 0,
                1.0 if index == 9 else -0.1,
                torch.tensor([float(index + 1)]), False,
            )
        )
    sample = buffer.sample(4, positive_fraction=0.25)
    assert sum(item.reward >= 0.5 for item in sample) == 1


def test_training_step_changes_weights() -> None:
    config = small_config()
    agent = Human.create("human_001", 1, 1, config)
    before = deepcopy(agent.brain.state_dict())
    for index in range(2):
        state = torch.full((Human.INPUT_SIZE,), float(index))
        next_state = state + 0.1
        agent.trainer.remember(state, index, 1.0, next_state, False)
    loss = agent.trainer.train_step(force=True)
    assert loss is not None
    assert any(not torch.equal(before[name], value) for name, value in agent.brain.state_dict().items())


def test_training_step_returns_finite_loss() -> None:
    config = small_config()
    agent = Animal.create("animal_001", 1, 1, config)
    for index in range(2):
        state = torch.full((Animal.INPUT_SIZE,), float(index))
        agent.trainer.remember(state, index, 0.5, state + 0.2, False)
    loss = agent.trainer.train_step(force=True)
    assert loss is not None and torch.isfinite(torch.tensor(loss))


def test_checkpoint_round_trip_preserves_outputs(tmp_path: Path) -> None:
    config = small_config()
    world = World(config)
    for agent in world.agents:
        state = torch.zeros(agent.input_size)
        agent.trainer.remember(state, 0, 0.25, state.clone(), False)
        agent.decision_steps = 7
        agent.children_born = 3
    checkpoint_dir = tmp_path / "checkpoints" / "experiment_001" / "run_001"
    initial_hashes = {agent.id: "initial" for agent in world.agents}
    save_run_checkpoints(world.agents, checkpoint_dir, 1, 1, 42, config, None, initial_hashes)
    probe_by_id = {
        agent.id: torch.randn(agent.input_size) for agent in world.agents
    }
    expected = {
        agent.id: agent.brain(probe_by_id[agent.id].unsqueeze(0)).detach()
        for agent in world.agents
    }
    positions = [(1, 1), (2, 2)]
    loaded, _, _ = load_agents(checkpoint_dir, config, positions)
    for agent in loaded:
        actual = agent.brain(probe_by_id[agent.id].unsqueeze(0)).detach()
        assert torch.equal(expected[agent.id], actual)
        assert len(agent.trainer.replay_buffer) == 1
        assert agent.decision_steps == 7
        original = next(item for item in world.agents if item.id == agent.id)
        assert agent.sex == original.sex
        assert agent.predator == original.predator
        assert agent.children_born == 3
        assert (checkpoint_dir / f"{agent.id}.pt").is_file()


def test_brain_v1_checkpoint_is_rejected_with_clear_migration_message(
    tmp_path: Path,
) -> None:
    config = small_config()
    world = World(config)
    checkpoint_dir = tmp_path / "checkpoints" / "experiment_001" / "run_001"
    save_run_checkpoints(world.agents, checkpoint_dir, 1, 1, 42, config, None, {})
    first_path = checkpoint_dir / "human_001.pt"
    payload = torch.load(first_path, map_location="cpu", weights_only=False)
    payload["architecture"]["architecture_version"] = 1
    torch.save(payload, first_path)

    with pytest.raises(CheckpointError, match="Brain v1 weights cannot be loaded"):
        load_agents(checkpoint_dir, config, [(1, 1), (2, 2)])


def test_checkpoint_files_and_metadata_are_saved(tmp_path: Path) -> None:
    config = small_config()
    world = World(config)
    checkpoint_dir = tmp_path / "experiment_001" / "run_001"
    save_run_checkpoints(world.agents, checkpoint_dir, 1, 1, 42, config, None, {})
    assert (checkpoint_dir / "metadata.json").is_file()
    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == [
        "animal_001.pt", "human_001.pt"
    ]


def test_reward_migration_preserves_weights_but_resets_learning_state(
    tmp_path: Path,
) -> None:
    old_config = small_config()
    old_config.reward_version = 1
    agent = Human.create("human_001", 1, 1, old_config)
    for index in range(2):
        state = torch.full((Human.INPUT_SIZE,), float(index))
        agent.trainer.remember(state, index, 1.0, state + 0.1, False)
    assert agent.trainer.train_step(force=True) is not None
    agent.decision_steps = 20_000
    checkpoint_dir = tmp_path / "experiment_001" / "run_001"
    old_horde = ReplayBuffer(10)
    old_horde.push(
        Experience(
            torch.zeros(Human.INPUT_SIZE), 0, 1.0,
            torch.ones(Human.INPUT_SIZE), False,
        )
    )
    save_run_checkpoints(
        [agent], checkpoint_dir, 1, 1, 42, old_config, None,
        {agent.id: model_hash(agent)},
        horde_replay_buffers={"human": old_horde, "animal": ReplayBuffer(10)},
    )
    probe = torch.randn(Human.INPUT_SIZE)
    expected = agent.brain(probe.unsqueeze(0)).detach()

    new_config = small_config()
    new_config.reward_version = 2
    loaded, _, _ = load_agents(checkpoint_dir, new_config, [(2, 2)])
    restored = loaded[0]

    assert torch.equal(expected, restored.brain(probe.unsqueeze(0)).detach())
    assert getattr(restored, "learning_state_reset") is True
    assert len(restored.trainer.replay_buffer) == 0
    assert restored.trainer.training_steps == 0
    assert not restored.trainer.optimizer.state
    restored.choose_action(torch.zeros(Human.INPUT_SIZE))
    assert restored.last_epsilon == restored.exploration_epsilon
    assert 0.01 <= restored.last_epsilon <= 0.50
    migrated_world = World(new_config, populate=False)
    migrated_world.agents = [restored]
    migrated_engine = SimulationEngine(
        migrated_world, new_config, 1, 2, 43, tmp_path,
        source_checkpoint=checkpoint_dir,
        learning_state_resets=[restored.id],
    )
    assert len(migrated_engine.horde_replay_buffers["human"]) == 0


def test_invalid_shape_and_out_of_bounds_are_detected() -> None:
    brain = AgentBrain(3, [2, 4, 4], 2, need_input_size=1)
    try:
        brain(torch.zeros(4))
    except ValueError as error:
        assert "Expected observation width" in str(error)
    else:
        raise AssertionError("Expected a shape validation error")

    config = small_config()
    world = World(config, populate=False)
    agent = Animal.create("animal_001", -1, 0, config)
    world.agents = [agent]
    try:
        world.validate()
    except RuntimeError as error:
        assert "escaped the grid" in str(error)
    else:
        raise AssertionError("Expected grid validation to fail")


def test_incremental_tick_feeds_every_living_agent_replay_buffer(tmp_path: Path) -> None:
    config = small_config()
    config.num_ticks = 3
    engine = SimulationEngine(World(config), config, 1, 1, 42, tmp_path)
    assert engine.current_tick == 0
    assert engine.step()
    assert engine.current_tick == 1
    assert all(len(agent.trainer.replay_buffer) == 1 for agent in engine.world.agents)
    assert all(agent.id in engine.last_agent_events for agent in engine.world.agents)


def test_web_controller_exposes_json_safe_brain_state(tmp_path: Path) -> None:
    config = small_config()
    config.num_ticks = 3
    engine = SimulationEngine(World(config), config, 1, 1, 42, tmp_path)
    controller = WebSimulationController(engine)
    controller.start()
    try:
        state = controller.control("step")
        assert state["tick"] == 1
        first_agent = state["agents"][0]
        assert len(first_agent["q_values"]) == len(Action)
        assert len(first_agent["observation"]) in {Human.INPUT_SIZE, Animal.INPUT_SIZE}
        assert first_agent["replay_size"] == 1
        assert set(first_agent["action_counts"]) == {action.name for action in Action}
        assert sum(first_agent["action_counts"].values()) == 1
        assert first_agent["brain_architecture"]["architecture_version"] == 2
        assert first_agent["brain_activations"]["fusion_hidden"]
        assert first_agent["weight_statistics"]
        assert first_agent["spatial_memory"].keys() == {"food", "water"}
        assert len(state["grid"]["terrain"]) == config.grid_height
        assert all(tree["variant"] in {0, 1, 2} for tree in state["grid"]["trees"])
        assert len(state["grid"]["houses"]) == 2
        assert all(
            house["required_materials"] == config.house_materials_required
            for house in state["grid"]["houses"]
        )
        assert "carried_wood" in first_agent
        assert state["summary"]["wood_remaining"] == len(state["grid"]["trees"])
        json.dumps(state, allow_nan=False)
    finally:
        controller.close()


def test_web_can_create_configured_population_and_brains(tmp_path: Path) -> None:
    config = small_config()
    config.num_ticks = 3
    first_engine = SimulationEngine(World(config), config, 20, 1, 42, tmp_path)
    received: list[SimulationConfig] = []

    def make_new(new_config: SimulationConfig, seed: int) -> SimulationEngine:
        received.append(new_config)
        return SimulationEngine(World(new_config), new_config, 21, 1, seed, tmp_path)

    controller = WebSimulationController(first_engine, new_engine_factory=make_new)
    controller.start()
    try:
        assert controller.state()["can_configure_experiment"]
        state = controller.control(
            "new_experiment",
            {
                "num_humans": 7,
                "num_animals": 13,
                "human_brain_width": 48,
                "animal_brain_width": 32,
            },
        )
        assert state["status"] == "paused"
        assert state["experiment_config"] == {
            "brain_architecture_version": 2,
            "horde_learning_enabled": True,
            "epsilon_standard_range": [0.01, 0.15],
            "epsilon_scout": 0.5,
            "num_humans": 7,
            "num_animals": 13,
            "food_per_agent": 1,
            "food_target": 20,
            "predator_fraction": 0.3,
            "human_hidden_sizes": [24, 48, 48],
            "animal_hidden_sizes": [16, 32, 32],
        }
        assert len(state["agents"]) == 20
        assert received[0].human_brain.hidden_sizes == [24, 48, 48]
        assert received[0].animal_brain.hidden_sizes == [16, 32, 32]
        controller.control("step")
        assert not controller.state()["can_configure_experiment"]
    finally:
        controller.close()


def test_web_can_cancel_before_first_tick_without_writing_checkpoint(
    tmp_path: Path,
) -> None:
    config = small_config()
    engine = SimulationEngine(World(config), config, 1, 1, 42, tmp_path)
    controller = WebSimulationController(engine)
    controller.start()
    try:
        assert controller.state()["can_cancel_experiment"]
        state = controller.control("cancel")
        assert state["status"] == "completed"
        assert state["termination_reason"] == "user_cancelled"
        assert state["result"] is None
        assert state["can_configure_experiment"]
        assert not state["can_start_next_run"]
        assert not (tmp_path / "checkpoints").exists()
    finally:
        controller.close()


def test_web_cancel_saves_an_advanced_run_as_partial_checkpoint(
    tmp_path: Path,
) -> None:
    config = small_config()
    config.num_ticks = 10
    config.generate_plots = False
    engine = SimulationEngine(World(config), config, 2, 1, 43, tmp_path)
    controller = WebSimulationController(engine)
    controller.start()
    try:
        controller.control("step")
        state = controller.control("cancel")
        assert state["status"] == "completed"
        assert state["tick"] == 1
        assert state["termination_reason"] == "user_cancelled"
        assert state["result"] is not None
        assert state["learning_summary"]["termination_reason"] == "user_cancelled"
        assert not state["can_start_next_run"]
        metadata = read_metadata(Path(state["result"]["checkpoint_dir"]))
        assert metadata["termination_reason"] == "user_cancelled"
        assert metadata["ticks_executed"] == 1
    finally:
        controller.close()


def test_brb_starts_new_population_from_champion_weights_only(tmp_path: Path) -> None:
    source_config = small_config()
    source_config.num_ticks = 1
    source_config.generate_plots = False
    source_config.compact_console = True
    source_engine = SimulationEngine(
        World(source_config), source_config, 1, 1, 42, tmp_path
    )
    source_engine.run()

    public = brb_public_summary(tmp_path)
    assert public is not None
    assert public["experiment_id"] == 1
    registry, payloads = load_brb_payloads(tmp_path)

    target_config = deepcopy(source_config)
    target_config.num_humans = 3
    target_config.num_animals = 2
    target_config.human_brain.hidden_sizes = [8, 16, 16]
    target_config.animal_brain.hidden_sizes = [8, 16, 16]
    engine = build_brb_engine(tmp_path, target_config, seed=43)

    assert engine.experiment_id == 2
    assert engine.run_number == 1
    assert engine.brb_source == registry["source_checkpoint"]
    assert len(engine.world.agents) == 5
    assert engine.horde_replay_buffers["human"].state_dict() == []
    assert all(len(agent.trainer.replay_buffer) == 0 for agent in engine.world.agents)
    assert all(not agent.trainer.optimizer.state for agent in engine.world.agents)
    assert set(engine.brb_parents) == {agent.id for agent in engine.world.agents}

    source_human = payloads["human"][0]
    cloned_human = next(
        agent for agent in engine.world.agents if agent.agent_type == "human"
    )
    source_brain = AgentBrain(
        input_size=source_human["architecture"]["input_size"],
        hidden_sizes=source_human["architecture"]["hidden_sizes"],
        output_size=source_human["architecture"]["output_size"],
        need_input_size=source_human["architecture"]["need_input_size"],
    )
    source_brain.load_state_dict(source_human["model_state_dict"])
    probe = torch.linspace(0, 1, Human.INPUT_SIZE)
    assert torch.allclose(source_brain(probe), cloned_human.brain(probe), atol=1e-6)


def test_brb_score_prioritizes_a_completed_run_with_human_survivors() -> None:
    def summary(reason: str, final_humans: int, mean_survival: float) -> dict[str, object]:
        return {
            "termination_reason": reason,
            "initial_humans": 5,
            "final_humans": final_humans,
            "by_type": {"human": {
                "mean_survival": mean_survival,
                "individual_survival": {f"human_{index}": int(mean_survival) for index in range(5)},
                "brain_selected_drinks": 10,
                "successful_meals": 10,
                "ignored_survival_priority_actions": 0,
            }},
        }

    completed = summary("tick_limit", final_humans=2, mean_survival=2_000.0)
    extinct = summary("human_extinction", final_humans=0, mean_survival=4_000.0)

    assert _score(completed) > _score(extinct)


def test_web_checkbox_routes_new_experiment_through_brb_factory(
    tmp_path: Path,
) -> None:
    config = small_config()
    first_engine = SimulationEngine(World(config), config, 1, 1, 42, tmp_path)
    calls: list[SimulationConfig] = []

    def make_fresh(new_config: SimulationConfig, seed: int) -> SimulationEngine:
        raise AssertionError("Fresh factory must not run when BRB is selected")

    def make_brb(new_config: SimulationConfig, seed: int) -> SimulationEngine:
        calls.append(new_config)
        return SimulationEngine(
            World(new_config), new_config, 2, 1, seed, tmp_path,
            brb_source="champion/run_050",
        )

    controller = WebSimulationController(
        first_engine,
        new_engine_factory=make_fresh,
        brb_engine_factory=make_brb,
        brb_summary_factory=lambda: {"experiment_id": 1, "run_number": 50},
    )
    controller.start()
    try:
        state = controller.control(
            "new_experiment",
            {
                "num_humans": 4,
                "num_animals": 6,
                "human_brain_width": 16,
                "animal_brain_width": 16,
                "use_brb": True,
            },
        )
        assert calls
        assert state["brb_source"] == "champion/run_050"
        assert state["brb"] == {"experiment_id": 1, "run_number": 50}
        assert len(state["agents"]) == 10
    finally:
        controller.close()


def test_run_stops_and_summarizes_when_last_human_dies(tmp_path: Path) -> None:
    config = small_config()
    config.num_ticks = 50
    world = World(config)
    human = next(agent for agent in world.agents if agent.agent_type == "human")
    human.health = 0.01
    human.hunger = 1.0
    human.thirst = 1.0
    world.food.clear()
    world.water.clear()
    source_checkpoint = tmp_path / "checkpoints" / "experiment_099" / "run_000"
    previous_results = tmp_path / "results" / "experiment_099" / "run_000"
    previous_results.mkdir(parents=True)
    (previous_results / "run_summary.json").write_text(
        json.dumps({
            "run_number": 0,
            "ticks_executed": 1,
            "configured_ticks": 50,
            "by_type": {"human": {"mean_survival": 0.5}},
        }),
        encoding="utf-8",
    )
    engine = SimulationEngine(
        world, config, 99, 1, 42, tmp_path, source_checkpoint=source_checkpoint
    )

    assert engine.step()
    assert engine.current_tick == 1
    assert engine.is_complete
    assert engine.termination_reason == "human_extinction"
    assert human.cause_of_death == "starvation+dehydration"
    assert not engine.step()

    result = engine.finalize()
    summary = engine.learning_summary
    assert summary is not None
    assert summary["ticks_executed"] == 1
    assert summary["initial_humans"] == 1
    assert summary["final_humans"] == 0
    assert summary["what_they_learned"]
    assert summary["by_type"]["human"]["deaths_with_critical_thirst"] == 1
    assert summary["comparison_to_previous_run"]["previous_last_human_tick"] == 1
    assert summary["comparison_to_previous_run"]["current_last_human_tick"] == 1
    metadata = read_metadata(result.checkpoint_dir)
    assert metadata["termination_reason"] == "human_extinction"
    assert metadata["ticks_executed"] == 1


def test_web_starts_next_run_with_exact_previous_weights(tmp_path: Path) -> None:
    config = small_config()
    config.num_ticks = 20
    world = World(config)
    human = next(agent for agent in world.agents if agent.agent_type == "human")
    human.health = 0.01
    human.hunger = human.thirst = 1.0
    world.food.clear()
    world.water.clear()
    first_engine = SimulationEngine(world, config, 7, 1, 100, tmp_path)

    def make_next(checkpoint: Path, next_config: SimulationConfig, seed: int) -> SimulationEngine:
        return build_resumed_engine(tmp_path, checkpoint, next_config, seed)

    controller = WebSimulationController(first_engine, next_engine_factory=make_next)
    controller.start()
    try:
        finished = controller.control("step")
        assert finished["status"] == "completed"
        assert finished["can_start_next_run"]
        previous_metadata = read_metadata(Path(finished["result"]["checkpoint_dir"]))

        continued = controller.control("next_run")
        controller.control("pause")
        assert continued["run_number"] == 2
        assert controller.engine.seed == 101
        assert controller.engine.initial_hashes == previous_metadata["final_model_hashes"]
        assert controller.engine.source_checkpoint == Path(
            finished["result"]["checkpoint_dir"]
        ).resolve()
    finally:
        controller.close()


def test_continuous_text_mode_chains_runs_and_checkpoints(
    tmp_path: Path, capsys: object
) -> None:
    config = small_config()
    config.num_ticks = 1
    config.status_every = 1
    config.compact_console = True
    config.generate_plots = False

    results = run_continuous(tmp_path, config, seed=70, max_runs=2)

    assert len(results) == 2
    assert results[0].checkpoint_dir.name == "run_001"
    assert results[1].checkpoint_dir.name == "run_002"
    first_metadata = read_metadata(results[0].checkpoint_dir)
    second_metadata = read_metadata(results[1].checkpoint_dir)
    assert first_metadata["horde_replay_file"] == "horde_replay.pt"
    assert (results[0].checkpoint_dir / "horde_replay.pt").is_file()
    first_horde = torch.load(
        results[0].checkpoint_dir / "horde_replay.pt",
        map_location="cpu",
        weights_only=False,
    )
    second_horde = torch.load(
        results[1].checkpoint_dir / "horde_replay.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert len(second_horde["human"]) > len(first_horde["human"])
    assert len(second_horde["animal"]) > len(first_horde["animal"])
    assert second_metadata["source_checkpoint"] == str(results[0].checkpoint_dir.resolve())
    assert second_metadata["initial_model_hashes"] == first_metadata["final_model_hashes"]
    assert not list(results[0].results_dir.glob("*.png"))
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "CYCLE START" in output
    assert "CYCLE END" in output
    assert "run anterior terminó en: tick 1" in output


def test_epsilon_is_an_individual_fixed_profile_not_a_global_decay() -> None:
    config = small_config()
    agent = Human.create("human_001", 1, 1, config)
    agent.decision_steps = 100
    state = torch.zeros(Human.INPUT_SIZE)
    agent.choose_action(state, global_step=500_000)
    assert agent.exploration_profile == "scout"
    assert agent.last_epsilon == 0.50
    assert agent.decision_steps == 101


def test_target_network_updates_on_configured_interval() -> None:
    config = small_config()
    config.human_brain.target_update_interval = 1
    agent = Human.create("human_001", 1, 1, config)
    for index in range(2):
        state = torch.full((Human.INPUT_SIZE,), float(index))
        agent.trainer.remember(state, index, 1.0, state + 0.1, False)
    assert agent.trainer.train_step(force=True) is not None
    for name, value in agent.brain.state_dict().items():
        assert torch.equal(value, agent.trainer.target_brain.state_dict()[name])


def test_narrow_checkpoint_migrates_to_wider_brain_without_changing_outputs(
    tmp_path: Path,
) -> None:
    old_config = small_config()
    world = World(old_config)
    checkpoint_dir = tmp_path / "experiment_010" / "run_001"
    save_run_checkpoints(world.agents, checkpoint_dir, 10, 1, 42, old_config, None, {})
    probes = {agent.id: torch.randn(agent.input_size) for agent in world.agents}
    expected = {
        agent.id: agent.brain(probes[agent.id].unsqueeze(0)).detach()
        for agent in world.agents
    }

    wider_config = small_config()
    wider_config.human_brain.hidden_sizes = [8, 16, 16]
    wider_config.animal_brain.hidden_sizes = [8, 12, 12]
    loaded, _, migrated = load_agents(
        checkpoint_dir, wider_config, [(1, 1), (2, 2)]
    )
    assert set(migrated) == {"human_001", "animal_001"}
    for agent in loaded:
        actual = agent.brain(probes[agent.id].unsqueeze(0)).detach()
        assert torch.allclose(expected[agent.id], actual, atol=1e-7, rtol=1e-6)


def test_brain_migration_can_add_perception_inputs_without_changing_old_output() -> None:
    old_brain = AgentBrain(
        input_size=10, hidden_sizes=[4, 8, 8], output_size=8, need_input_size=4
    )
    new_brain = AgentBrain(
        input_size=12, hidden_sizes=[8, 16, 16], output_size=8, need_input_size=4
    )
    old_input = torch.randn(1, 10)
    expected = old_brain(old_input).detach()
    _load_widened_state_dict(new_brain, old_brain.state_dict())
    migrated_input = torch.nn.functional.pad(old_input, (0, 2))
    actual = new_brain(migrated_input).detach()
    assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-6)


def test_brain_migration_can_add_attack_output_without_changing_old_outputs() -> None:
    old_brain = AgentBrain(
        input_size=10, hidden_sizes=[4, 8, 8], output_size=8, need_input_size=4
    )
    new_brain = AgentBrain(
        input_size=10, hidden_sizes=[4, 8, 8], output_size=9, need_input_size=4
    )
    probe = torch.randn(1, 10)
    expected = old_brain(probe).detach()
    _load_widened_state_dict(new_brain, old_brain.state_dict())
    actual = new_brain(probe).detach()
    assert torch.allclose(expected, actual[:, :8], atol=1e-7, rtol=1e-6)


def test_house_inputs_and_build_action_preserve_previous_brain_outputs() -> None:
    old_brain = AgentBrain(
        input_size=44, hidden_sizes=[16, 32, 32], output_size=11,
        need_input_size=15,
    )
    new_brain = AgentBrain(
        input_size=Human.INPUT_SIZE, hidden_sizes=[16, 32, 32],
        output_size=len(Action), need_input_size=Human.NEED_INPUT_SIZE,
    )
    old_input = torch.randn(1, 44)
    expected = old_brain(old_input).detach()
    _load_widened_state_dict(new_brain, old_brain.state_dict())
    migrated_input = torch.nn.functional.pad(old_input, (0, Human.INPUT_SIZE - 44))
    actual = new_brain(migrated_input).detach()
    assert torch.allclose(expected, actual[:, :11], atol=1e-7, rtol=1e-6)


def test_brain_migration_can_insert_survival_inputs_before_spatial_branch() -> None:
    old_brain = AgentBrain(
        input_size=26, hidden_sizes=[8, 12, 12], output_size=8,
        need_input_size=8,
    )
    new_brain = AgentBrain(
        input_size=33, hidden_sizes=[8, 12, 12], output_size=8,
        need_input_size=15,
    )
    old_input = torch.randn(1, 26)
    expected = old_brain(old_input).detach()
    _load_widened_state_dict(new_brain, old_brain.state_dict())
    migrated_input = torch.cat(
        (old_input[:, :8], torch.zeros(1, 7), old_input[:, 8:]), dim=1
    )
    actual = new_brain(migrated_input).detach()
    assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-6)

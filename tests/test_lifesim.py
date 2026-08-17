from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import torch
import pytest

from agents.animal import Animal
from agents.brain import AgentBrain
from agents.human import Human
from config import BrainConfig, SimulationConfig
from learning.replay_buffer import Experience, ReplayBuffer
from learning.rewards import calculate_reward
from persistence.checkpoints import (
    CheckpointError,
    _load_widened_state_dict,
    load_agents,
    model_hash,
    read_metadata,
    save_run_checkpoints,
)
from simulation.engine import SimulationEngine
from simulation.experiment import build_resumed_engine, run_continuous
from web.server import WebSimulationController
from world.grid import Action
from world.world import World


def small_config() -> SimulationConfig:
    return SimulationConfig(
        grid_width=8,
        grid_height=6,
        num_humans=1,
        num_animals=1,
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
    engine.step()
    assert len(humans[0].trainer.learning_replay_buffer) == 4
    assert all(agent.trainer.training_steps > 0 for agent in humans)


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


def test_food_regrows_next_to_existing_food() -> None:
    config = small_config()
    config.food_respawn_probability = 1.0
    world = World(config, populate=False)
    world.food.add((3, 3))
    world.update_resources()
    assert len(world.food) == 2
    assert any(abs(x - 3) + abs(y - 3) == 1 for x, y in world.food)


def test_default_world_uses_dense_matrix() -> None:
    config = SimulationConfig()
    assert (config.grid_width, config.grid_height) == (60, 40)


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


def test_brain_v2_prioritizes_needs_at_fifty_percent() -> None:
    config = small_config()
    world = World(config, populate=False)
    agent = Human.create("human_001", 2, 2, config)
    world.agents = [agent]
    agent.hunger = 0.50
    agent.thirst = 0.49
    agent.energy = 0.50
    agent.health = 0.49
    perception = agent.perceive(world)
    assert perception[4:8].tolist() == [1.0, 0.0, 1.0, 1.0]


def test_spatial_memory_keeps_last_seen_water_outside_local_vision() -> None:
    config = small_config()
    config.vision_radius = 1
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
    save_run_checkpoints(
        [agent], checkpoint_dir, 1, 1, 42, old_config, None,
        {agent.id: model_hash(agent)},
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
        assert first_agent["brain_architecture"]["architecture_version"] == 2
        assert first_agent["brain_activations"]["fusion_hidden"]
        assert first_agent["weight_statistics"]
        assert first_agent["spatial_memory"].keys() == {"food", "water"}
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

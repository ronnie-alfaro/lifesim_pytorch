from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from agents.animal import Animal, predator_for_agent_id
from agents.base_agent import BaseAgent, sex_for_agent_id
from agents.human import Human
from config import BrainConfig, SimulationConfig
from learning.replay_buffer import ReplayBuffer
from world.grid import Action


class CheckpointError(RuntimeError):
    pass


def model_hash(agent: BaseAgent) -> str:
    return state_dict_hash(agent.brain.state_dict())


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_run_checkpoints(
    agents: list[BaseAgent], checkpoint_dir: Path, experiment_id: int, run_number: int,
    seed: int, config: SimulationConfig, source_checkpoint: str | None,
    initial_hashes: dict[str, str], termination_reason: str | None = None,
    ticks_executed: int | None = None,
    horde_replay_buffers: dict[str, ReplayBuffer] | None = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent_files: list[str] = []
    final_hashes: dict[str, str] = {}
    for agent in agents:
        path = checkpoint_dir / f"{agent.id}.pt"
        payload = {
            "format_version": 1,
            "reward_version": config.reward_version,
            "agent_id": agent.id,
            "agent_type": agent.agent_type,
            "sex": agent.sex,
            "predator": agent.predator,
            "cause_of_death": agent.cause_of_death,
            "carried_food": agent.carried_food,
            "heart_partner_id": agent.heart_partner_id,
            "heart_ticks_remaining": agent.heart_ticks_remaining,
            "pregnant_by_id": agent.pregnant_by_id,
            "pregnancy_ticks_remaining": agent.pregnancy_ticks_remaining,
            "mother_id": agent.mother_id,
            "dependent_ticks_remaining": agent.dependent_ticks_remaining,
            "dependent_ids": list(agent.dependent_ids),
            "children_born": agent.children_born,
            "architecture": agent.brain.architecture,
            "learning_rate": agent.brain_config.learning_rate,
            "model_state_dict": agent.brain.state_dict(),
            "optimizer_state_dict": agent.trainer.optimizer.state_dict(),
            "target_model_state_dict": agent.trainer.target_brain.state_dict(),
            "replay_buffer": agent.trainer.replay_buffer.state_dict(),
            "remembered_steps": agent.trainer.remembered_steps,
            "training_steps": agent.trainer.training_steps,
            "decision_steps": agent.decision_steps,
            "exploration_profile": agent.exploration_profile,
            "exploration_epsilon": agent.exploration_epsilon,
            "total_reward": agent.total_reward,
            "steps_survived": agent.steps_survived,
            "run": run_number,
            "experiment_id": experiment_id,
        }
        torch.save(payload, path)
        agent_files.append(path.name)
        final_hashes[agent.id] = model_hash(agent)
    horde_replay_file = None
    horde_replay_hash = None
    if horde_replay_buffers:
        horde_path = checkpoint_dir / "horde_replay.pt"
        torch.save(
            {
                agent_type: replay.state_dict()
                for agent_type, replay in horde_replay_buffers.items()
            },
            horde_path,
        )
        horde_replay_file = horde_path.name
        horde_replay_hash = hashlib.sha256(horde_path.read_bytes()).hexdigest()
    metadata = {
        "format_version": 1,
        "experiment_id": experiment_id,
        "run_number": run_number,
        "seed": seed,
        "source_checkpoint": source_checkpoint,
        "termination_reason": termination_reason,
        "ticks_executed": ticks_executed,
        "config": config.to_dict(),
        "agent_files": agent_files,
        "initial_model_hashes": initial_hashes,
        "final_model_hashes": final_hashes,
        "horde_replay_file": horde_replay_file,
        "horde_replay_hash": horde_replay_hash,
        "horde_observation_schema": {
            agent_type: next(
                agent.brain.architecture
                for agent in agents
                if agent.agent_type == agent_type
            )
            for agent_type in ("human", "animal")
            if any(agent.agent_type == agent_type for agent in agents)
        },
    }
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return checkpoint_dir


def read_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    path = checkpoint_dir / "metadata.json"
    if not path.is_file():
        raise CheckpointError(f"Checkpoint metadata not found: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"Cannot read checkpoint metadata: {error}") from error
    if metadata.get("format_version") != 1:
        raise CheckpointError("Unsupported checkpoint format version")
    return metadata


def load_agents(
    checkpoint_dir: Path, config: SimulationConfig, positions: list[tuple[int, int]]
) -> tuple[list[BaseAgent], dict[str, str], list[str]]:
    metadata = read_metadata(checkpoint_dir)
    files = metadata.get("agent_files")
    if not isinstance(files, list) or len(files) != len(positions):
        raise CheckpointError("Checkpoint agent count is incompatible with the new world")
    loaded_agents: list[BaseAgent] = []
    hashes: dict[str, str] = {}
    migrated_agents: list[str] = []
    for filename, (x, y) in zip(files, positions):
        payload = torch.load(checkpoint_dir / filename, map_location="cpu", weights_only=False)
        required = {"agent_id", "agent_type", "architecture", "model_state_dict"}
        if not required.issubset(payload):
            raise CheckpointError(f"Incomplete agent checkpoint: {filename}")
        architecture = payload["architecture"]
        architecture_version = int(architecture.get("architecture_version", 1))
        if architecture_version != config.brain_architecture_version:
            raise CheckpointError(
                f"Incompatible brain architecture in {filename}: checkpoint v"
                f"{architecture_version}, current v{config.brain_architecture_version}. "
                "Brain v1 weights cannot be loaded into Brain v2; start a new "
                "experiment with --new."
            )
        reward_changed = int(payload.get("reward_version", 1)) != config.reward_version
        configured_brain = (
            config.human_brain if payload["agent_type"] == "human" else config.animal_brain
        )
        old_hidden_sizes = list(architecture["hidden_sizes"])
        old_need_input = int(architecture.get("need_input_size", 0))
        configured_hidden_sizes = list(configured_brain.hidden_sizes)
        if len(old_hidden_sizes) == len(configured_hidden_sizes):
            hidden_sizes = [
                max(old_size, configured_size)
                for old_size, configured_size in zip(old_hidden_sizes, configured_hidden_sizes)
            ]
        else:
            hidden_sizes = old_hidden_sizes
        brain_config = BrainConfig(
            hidden_sizes=hidden_sizes,
            learning_rate=float(payload.get("learning_rate", 0.001)),
            replay_capacity=configured_brain.replay_capacity,
            batch_size=configured_brain.batch_size,
            gamma=configured_brain.gamma,
            train_every=configured_brain.train_every,
            target_update_interval=configured_brain.target_update_interval,
            positive_replay_fraction=configured_brain.positive_replay_fraction,
        )
        expected_input = Human.INPUT_SIZE if payload["agent_type"] == "human" else Animal.INPUT_SIZE
        expected_need_input = (
            Human.NEED_INPUT_SIZE
            if payload["agent_type"] == "human"
            else Animal.NEED_INPUT_SIZE
        )
        if (
            architecture["input_size"] > expected_input
            or old_need_input > expected_need_input
            or old_need_input <= 0
            or architecture["output_size"] > len(Action)
        ):
            raise CheckpointError(f"Incompatible architecture in {filename}: {architecture}")
        agent_class = Human if payload["agent_type"] == "human" else Animal
        agent = agent_class(
            payload["agent_id"], x, y, config, brain_config,
            expected_input, payload["agent_type"],
            sex=str(payload.get("sex", sex_for_agent_id(payload["agent_id"]))),
            predator=bool(payload.get(
                "predator",
                predator_for_agent_id(payload["agent_id"], config)
                if payload["agent_type"] == "animal" else False,
            )),
            children_born=int(payload.get("children_born", 0)),
        )
        expected_hash = metadata.get("final_model_hashes", {}).get(agent.id)
        source_hash = state_dict_hash(payload["model_state_dict"])
        if expected_hash and source_hash != expected_hash:
            raise CheckpointError(f"Weight integrity check failed for {agent.id}")
        migrated = (
            hidden_sizes != old_hidden_sizes
            or architecture["input_size"] != expected_input
            or old_need_input != expected_need_input
            or architecture["output_size"] != len(Action)
        )
        try:
            if migrated:
                _load_widened_state_dict(agent.brain, payload["model_state_dict"])
                migrated_agents.append(agent.id)
            else:
                agent.brain.load_state_dict(payload["model_state_dict"], strict=True)
            target_state = payload.get("target_model_state_dict", payload["model_state_dict"])
            if reward_changed:
                agent.trainer.target_brain.load_state_dict(agent.brain.state_dict())
            elif migrated:
                _load_widened_state_dict(agent.trainer.target_brain, target_state)
            else:
                agent.trainer.target_brain.load_state_dict(target_state, strict=True)
            if "optimizer_state_dict" in payload and not migrated and not reward_changed:
                agent.trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "replay_buffer" in payload and not reward_changed:
                agent.trainer.replay_buffer.load_state_dict(
                    payload["replay_buffer"], expected_input,
                    source_need_input_size=old_need_input,
                    target_need_input_size=expected_need_input,
                )
            if reward_changed:
                agent.trainer.remembered_steps = 0
                agent.trainer.training_steps = 0
                agent.decision_steps = 0
                agent.learning_state_reset = True
            else:
                agent.trainer.remembered_steps = int(
                    payload.get("remembered_steps", len(agent.trainer.replay_buffer))
                )
                agent.trainer.training_steps = int(payload.get("training_steps", 0))
                agent.decision_steps = int(payload.get("decision_steps", 0))
                agent.learning_state_reset = False
            saved_epsilon = payload.get("exploration_epsilon")
            if saved_epsilon is not None:
                epsilon = float(saved_epsilon)
                if not 0.0 <= epsilon <= config.epsilon_scout:
                    raise ValueError(f"Invalid individual epsilon {epsilon}")
                agent.exploration_epsilon = epsilon
                agent.exploration_profile = str(
                    payload.get("exploration_profile", "standard")
                )
                agent.last_epsilon = epsilon
        except (RuntimeError, ValueError) as error:
            raise CheckpointError(f"Cannot reconstruct {filename}: {error}") from error
        hashes[agent.id] = model_hash(agent)
        if expected_hash and not migrated and hashes[agent.id] != expected_hash:
            raise CheckpointError(f"Weight integrity check failed for {agent.id}")
        loaded_agents.append(agent)
    return loaded_agents, hashes, migrated_agents


def load_horde_replay_state(
    checkpoint_dir: Path,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, dict[str, object]],
] | None:
    """Load and integrity-check the persisted species-wide replay, if present."""
    metadata = read_metadata(checkpoint_dir)
    filename = metadata.get("horde_replay_file")
    if not filename:
        return None
    path = checkpoint_dir / str(filename)
    if not path.is_file():
        raise CheckpointError(f"Horde replay not found: {path}")
    expected_hash = metadata.get("horde_replay_hash")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise CheckpointError("Horde replay integrity check failed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("Horde replay checkpoint must be a dictionary")
    for agent_type in ("human", "animal"):
        if agent_type not in payload or not isinstance(payload[agent_type], list):
            raise CheckpointError(f"Horde replay is missing {agent_type} experiences")
    schemas = metadata.get("horde_observation_schema")
    if not isinstance(schemas, dict):
        schemas = {}
        for filename in metadata.get("agent_files", []):
            agent_payload = torch.load(
                checkpoint_dir / filename, map_location="cpu", weights_only=False
            )
            agent_type = str(agent_payload.get("agent_type"))
            if agent_type not in schemas:
                architecture = agent_payload.get("architecture")
                if isinstance(architecture, dict):
                    schemas[agent_type] = architecture
    return payload, schemas


def _load_widened_state_dict(
    model: torch.nn.Module, source_state: dict[str, torch.Tensor]
) -> None:
    """Embed a narrower MLP in a wider one while preserving its initial outputs."""
    target_state = model.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError("Cannot migrate brains with a different number of layers")
    for key, source in source_state.items():
        target = target_state[key].clone()
        if source.shape == target.shape:
            target_state[key] = source
            continue
        if source.ndim == 2 and target.ndim == 2:
            old_out, old_in = source.shape
            new_out, new_in = target.shape
            if old_out > new_out or old_in > new_in:
                raise ValueError(f"Cannot shrink layer {key} from {source.shape} to {target.shape}")
            if key == "fusion.0.weight":
                # Concatenation is [needs, spatial]. Widening the needs branch
                # moves the beginning of the spatial block to a new offset.
                old_need = source_state["need_branch.0.bias"].shape[0]
                new_need = target_state["need_branch.0.bias"].shape[0]
                old_spatial = old_in - old_need
                target[:old_out, :] = 0.0
                target[:old_out, :old_need] = source[:, :old_need]
                target[
                    :old_out, new_need : new_need + old_spatial
                ] = source[:, old_need:]
                target_state[key] = target
                continue
            target[:old_out, :old_in] = source
            # Existing neurons must not receive signals from newly added neurons.
            if new_in > old_in:
                target[:old_out, old_in:] = 0.0
            target_state[key] = target
        elif source.ndim == 1 and target.ndim == 1:
            if source.shape[0] > target.shape[0]:
                raise ValueError(f"Cannot shrink bias {key}")
            target[: source.shape[0]] = source
            target_state[key] = target
        else:
            raise ValueError(f"Unsupported migration for tensor {key}")
    model.load_state_dict(target_state, strict=True)

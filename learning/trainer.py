"""Pedagogical one-network DQN-style trainer."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import nn

from agents.brain import AgentBrain
from config import BrainConfig
from learning.replay_buffer import Experience, ReplayBuffer


class DQNTrainer:
    def __init__(self, brain: AgentBrain, config: BrainConfig) -> None:
        self.brain = brain
        self.target_brain = deepcopy(brain)
        self.target_brain.eval()
        self.config = config
        self.optimizer = torch.optim.Adam(brain.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)
        self.horde_replay_buffer: ReplayBuffer | None = None
        self.loss_function = nn.SmoothL1Loss()
        self.remembered_steps = 0
        self.training_steps = 0
        self.last_loss: float | None = None

    def remember(
        self, state: torch.Tensor, action: int, reward: float,
        next_state: torch.Tensor, done: bool,
        next_action_mask: torch.Tensor | None = None,
    ) -> None:
        if state.shape != next_state.shape or state.numel() != self.brain.input_size:
            raise ValueError("Experience state shapes do not match the brain input")
        if next_action_mask is not None:
            if next_action_mask.shape != (self.brain.output_size,):
                raise ValueError("Next-action mask width does not match brain outputs")
            if not next_action_mask.any():
                raise ValueError("Next-action mask cannot disable every action")
            next_action_mask = next_action_mask.detach().clone().to(dtype=torch.bool)
        experience = Experience(
            state.detach().clone(), action, reward, next_state.detach().clone(), done,
            next_action_mask,
        )
        self.replay_buffer.push(experience)
        if self.horde_replay_buffer is not None:
            self.horde_replay_buffer.push(experience)
        self.remembered_steps += 1

    def join_horde(self, replay_buffer: ReplayBuffer) -> None:
        """Train from a species-wide replay while retaining personal history."""
        self.horde_replay_buffer = replay_buffer

    @property
    def learning_replay_buffer(self) -> ReplayBuffer:
        return (
            self.horde_replay_buffer
            if self.horde_replay_buffer is not None
            else self.replay_buffer
        )

    def train_step(self, force: bool = False) -> float | None:
        learning_buffer = self.learning_replay_buffer
        if len(learning_buffer) < self.config.batch_size:
            return None
        if not force and self.remembered_steps % self.config.train_every != 0:
            return None
        batch = learning_buffer.sample(
            self.config.batch_size, self.config.positive_replay_fraction
        )
        states = torch.stack([item.state for item in batch])
        actions = torch.tensor([item.action for item in batch], dtype=torch.long)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
        next_states = torch.stack([item.next_state for item in batch])
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32)
        next_action_masks = torch.stack([
            item.next_action_mask
            if item.next_action_mask is not None
            else torch.ones(self.brain.output_size, dtype=torch.bool)
            for item in batch
        ])

        # Forward pass: select Q(s, action) for the actions actually taken.
        self.brain.train()
        q_values = self.brain(states)
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bootstrap target. detach/no_grad prevents target values from receiving gradients.
        with torch.no_grad():
            next_q_values = self.target_brain(next_states)
            next_q_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
            best_next_q_values = next_q_values.max(dim=1).values
            targets = rewards + self.config.gamma * best_next_q_values * (1.0 - dones)

        loss = self.loss_function(selected_q_values, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        # This is where weights change: clear gradients -> backward -> optimizer step.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())
            self.target_brain.eval()
        for name, parameter in self.brain.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite weights after update in {name}")
        self.last_loss = float(loss.item())
        if not math.isfinite(self.last_loss):
            raise FloatingPointError("Loss converted to a non-finite Python value")
        return self.last_loss

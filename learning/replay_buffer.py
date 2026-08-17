from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    next_action_mask: torch.Tensor | None = None


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._items: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self._items.append(experience)

    def sample(
        self, batch_size: int, positive_fraction: float = 0.0
    ) -> list[Experience]:
        if not self._items:
            raise RuntimeError("Cannot sample an empty replay buffer")
        if batch_size > len(self._items):
            raise ValueError("Batch size exceeds replay buffer length")
        items = list(self._items)
        if positive_fraction <= 0.0:
            return random.sample(items, batch_size)
        positive_items = [item for item in items if item.reward >= 0.5]
        positive_count = min(
            len(positive_items), max(0, round(batch_size * positive_fraction))
        )
        selected_positive = random.sample(positive_items, positive_count)
        selected_ids = {id(item) for item in selected_positive}
        remaining = [item for item in items if id(item) not in selected_ids]
        return selected_positive + random.sample(remaining, batch_size - positive_count)

    def __len__(self) -> int:
        return len(self._items)

    def state_dict(self) -> list[dict[str, object]]:
        return [
            {
                "state": item.state.cpu(),
                "action": item.action,
                "reward": item.reward,
                "next_state": item.next_state.cpu(),
                "done": item.done,
                "next_action_mask": (
                    item.next_action_mask.cpu()
                    if item.next_action_mask is not None
                    else None
                ),
            }
            for item in self._items
        ]

    def load_state_dict(
        self,
        items: list[dict[str, object]],
        input_size: int,
        source_need_input_size: int | None = None,
        target_need_input_size: int | None = None,
    ) -> None:
        self._items.clear()
        for index, item in enumerate(items[-self._items.maxlen :]):
            state = item.get("state")
            next_state = item.get("next_state")
            if not isinstance(state, torch.Tensor) or not isinstance(next_state, torch.Tensor):
                raise ValueError(f"Replay experience {index} has invalid tensors")
            if state.numel() > input_size or next_state.numel() > input_size:
                raise ValueError(f"Replay experience {index} is incompatible with input size")
            if state.numel() < input_size:
                state = _expand_observation(
                    state, input_size, source_need_input_size, target_need_input_size
                )
                next_state = _expand_observation(
                    next_state, input_size,
                    source_need_input_size, target_need_input_size,
                )
            self.push(
                Experience(
                    state.detach().clone().to(dtype=torch.float32),
                    int(item["action"]),
                    float(item["reward"]),
                    next_state.detach().clone().to(dtype=torch.float32),
                    bool(item["done"]),
                    _load_action_mask(item.get("next_action_mask"), index),
                )
            )


def _load_action_mask(value: object, index: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"Replay experience {index} has an invalid action mask")
    mask = value.detach().clone().to(dtype=torch.bool)
    if not mask.any():
        raise ValueError(f"Replay experience {index} masks every action")
    return mask


def _expand_observation(
    state: torch.Tensor,
    target_size: int,
    source_need_size: int | None,
    target_need_size: int | None,
) -> torch.Tensor:
    """Insert new need inputs before spatial inputs, then append other additions."""
    state = state.flatten().to(dtype=torch.float32)
    if (
        source_need_size is not None
        and target_need_size is not None
        and target_need_size >= source_need_size
        and source_need_size <= state.numel()
    ):
        need_padding = torch.zeros(target_need_size - source_need_size)
        state = torch.cat(
            (state[:source_need_size], need_padding, state[source_need_size:])
        )
    if state.numel() > target_size:
        raise ValueError("Expanded replay observation exceeds target input size")
    return torch.nn.functional.pad(state, (0, target_size - state.numel()))

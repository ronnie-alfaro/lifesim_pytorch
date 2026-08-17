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
            }
            for item in self._items
        ]

    def load_state_dict(self, items: list[dict[str, object]], input_size: int) -> None:
        self._items.clear()
        for index, item in enumerate(items[-self._items.maxlen :]):
            state = item.get("state")
            next_state = item.get("next_state")
            if not isinstance(state, torch.Tensor) or not isinstance(next_state, torch.Tensor):
                raise ValueError(f"Replay experience {index} has invalid tensors")
            if state.numel() > input_size or next_state.numel() > input_size:
                raise ValueError(f"Replay experience {index} is incompatible with input size")
            if state.numel() < input_size:
                state = torch.nn.functional.pad(state.flatten(), (0, input_size - state.numel()))
                next_state = torch.nn.functional.pad(
                    next_state.flatten(), (0, input_size - next_state.numel())
                )
            self.push(
                Experience(
                    state.detach().clone().to(dtype=torch.float32),
                    int(item["action"]),
                    float(item["reward"]),
                    next_state.detach().clone().to(dtype=torch.float32),
                    bool(item["done"]),
                )
            )

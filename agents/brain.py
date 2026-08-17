"""Brain v2: separate need/spatial encoders followed by a decision head."""

from __future__ import annotations

import torch
from torch import nn


class AgentBrain(nn.Module):
    ARCHITECTURE_VERSION = 2

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        need_input_size: int,
    ) -> None:
        super().__init__()
        if (
            input_size <= 0
            or output_size <= 0
            or not 0 < need_input_size < input_size
            or len(hidden_sizes) != 3
            or any(size <= 0 for size in hidden_sizes)
        ):
            raise ValueError("All brain layer sizes must be positive")
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.output_size = output_size
        self.need_input_size = need_input_size
        self.spatial_input_size = input_size - need_input_size
        need_hidden, spatial_hidden, fusion_hidden = hidden_sizes
        self.need_branch = nn.Sequential(
            nn.Linear(need_input_size, need_hidden), nn.ReLU()
        )
        self.spatial_branch = nn.Sequential(
            nn.Linear(self.spatial_input_size, spatial_hidden), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(need_hidden + spatial_hidden, fusion_hidden), nn.ReLU()
        )
        self.action_head = nn.Linear(fusion_hidden, output_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected observation width {self.input_size}, got {observations.shape[-1]}"
            )
        q_values, _ = self.forward_with_activations(observations)
        return q_values

    def forward_with_activations(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return Q-values plus interpretable activations for the web laboratory."""
        if observations.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected observation width {self.input_size}, got {observations.shape[-1]}"
            )
        needs = observations[..., : self.need_input_size]
        spatial = observations[..., self.need_input_size :]
        need_hidden = self.need_branch(needs)
        spatial_hidden = self.spatial_branch(spatial)
        fused = self.fusion(torch.cat((need_hidden, spatial_hidden), dim=-1))
        q_values = self.action_head(fused)
        return q_values, {
            "need_inputs": needs,
            "need_hidden": need_hidden,
            "spatial_inputs": spatial,
            "spatial_hidden": spatial_hidden,
            "fusion_hidden": fused,
            "q_values": q_values,
        }

    def weight_statistics(self) -> list[dict[str, float | str]]:
        """Small summaries let the UI show learning without sending full matrices."""
        rows: list[dict[str, float | str]] = []
        for name, parameter in self.named_parameters():
            if not name.endswith("weight"):
                continue
            rows.append({
                "layer": name.removesuffix(".weight"),
                "mean_abs": float(parameter.detach().abs().mean().item()),
                "max_abs": float(parameter.detach().abs().max().item()),
            })
        return rows

    @property
    def architecture(self) -> dict[str, int | list[int]]:
        return {
            "architecture_version": self.ARCHITECTURE_VERSION,
            "input_size": self.input_size,
            "need_input_size": self.need_input_size,
            "spatial_input_size": self.spatial_input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
        }

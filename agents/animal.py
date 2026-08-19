from __future__ import annotations

import math

import torch

from agents.base_agent import BaseAgent, sex_for_agent_id


class Animal(BaseAgent):
    """Brain v2 perception: survival signals followed by spatial memory."""

    NEED_LABELS = [
        "hambre", "sed", "falta energía", "falta salud",
        "prioridad comida", "prioridad agua", "prioridad descanso",
        "prioridad salud",
        "riesgo hambre", "riesgo sed", "riesgo agotamiento",
        "recibiendo daño", "daño reciente", "margen de vida",
        "urgencia vital",
    ]
    SPATIAL_LABELS = [
        "mem comida dx", "mem comida dy", "conf comida", "edad comida",
        "mem agua dx", "mem agua dy", "conf agua", "edad agua",
        "obstáculo arriba", "obstáculo abajo", "obstáculo izquierda",
        "obstáculo derecha", "x", "y", "comida al alcance", "agua al alcance",
        "comida cargada", "reserva dx", "reserva dy", "comida reserva",
        "reserva al alcance", "pareja al alcance", "corazón activo",
        "embarazo", "cuidando bebés", "hambre máxima bebés",
        "es bebé dependiente",
    ]
    NEED_INPUT_SIZE = len(NEED_LABELS)
    INPUT_SIZE = NEED_INPUT_SIZE + len(SPATIAL_LABELS)

    def perceive(self, world: "World") -> torch.Tensor:
        self.spatial_memory.observe(self, world)
        needs = self.need_observation()
        spatial = [
            *self.spatial_memory.features("food", self, world),
            *self.spatial_memory.features("water", self, world),
            *world.cardinal_obstacle_flags(self.x, self.y),
            self.x / max(1, world.width - 1),
            self.y / max(1, world.height - 1),
            float(world.resource_in_reach(self, "food")),
            float(world.resource_in_reach(self, "water")),
            *self.social_observation(world),
        ]
        return torch.tensor([*needs, *spatial], dtype=torch.float32)

    @classmethod
    def create(cls, agent_id: str, x: int, y: int, config: "SimulationConfig") -> "Animal":
        return cls(
            agent_id, x, y, config, config.animal_brain, cls.INPUT_SIZE, "animal",
            sex=sex_for_agent_id(agent_id),
            predator=predator_for_agent_id(agent_id, config),
        )


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import SimulationConfig
    from world.world import World


def predator_for_agent_id(agent_id: str, config: "SimulationConfig") -> bool:
    try:
        index = int(agent_id.rsplit("_", 1)[-1])
    except ValueError:
        index = 1
    predator_count = math.ceil(config.num_animals * config.predator_fraction)
    return index <= predator_count

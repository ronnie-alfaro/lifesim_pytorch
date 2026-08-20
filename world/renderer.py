from __future__ import annotations

from world.world import World


class AsciiRenderer:
    """Optional view kept entirely separate from simulation state changes."""

    def render(self, world: World) -> str:
        cells = [["." for _ in range(world.width)] for _ in range(world.height)]
        for x, y in world.obstacles:
            cells[y][x] = "#"
        for x, y in world.food:
            cells[y][x] = "F"
        for x, y in world.water:
            cells[y][x] = "W"
        for stockpile in world.stockpiles.values():
            cells[stockpile.y][stockpile.x] = "S"
        for house in world.houses.values():
            if house.complete:
                cells[house.y][house.x] = "C"
        for agent in world.living_agents:
            cells[agent.y][agent.x] = "H" if agent.agent_type == "human" else "A"
        return "\n".join("".join(row) for row in cells)

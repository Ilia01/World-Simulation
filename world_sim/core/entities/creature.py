from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..behaviors.movement import MovementBehavior
from ..behaviors.senses import SensingBehavior
from ..behaviors.reproduction import ReproductionBehavior
from .traits import Traits


class Creature:
    def __init__(
        self,
        x: int,
        y: int,
        movement_strategy: Optional[MovementBehavior] = None,
        sensing_strategy: Optional[SensingBehavior] = None,
        reproduction_strategy: Optional[ReproductionBehavior] = None,
        traits: Optional[Traits] = None,
        energy: float = 10.0,
        lineage: Optional[int] = None,
    ):
        self.x = x
        self.y = y
        self.energy = energy
        self.age = 0
        self.traits = traits or Traits()
        self.ate_food = False
        self.energy_gain = 0.0
        self.lineage = lineage

        self._movement = movement_strategy or MovementBehavior()
        self._sensing = sensing_strategy or SensingBehavior()
        self._reproduction = reproduction_strategy or ReproductionBehavior()

    def step(self, world_grid: np.ndarray, world_size: int, rng: np.random.Generator) -> None:
        target = self._sensing.find_closest_food(self, world_grid, rng)
        self._movement.move(self, target, world_size, rng)
        self.eat(world_grid)
        self.age += 1
        self.energy -= self.traits.metabolism
        # Natural death by age
        if self.age >= getattr(self.traits, "max_age", 200):
            self.energy = 0.0

    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def is_alive(self) -> bool:
        return self.energy > 0

    def eat(self, world_grid: np.ndarray, food_energy: float = 5.0) -> None:
        self.ate_food = False
        self.energy_gain = 0.0
        if world_grid[self.x, self.y] == "F":
            self.energy += food_energy
            world_grid[self.x, self.y] = None
            self.ate_food = True
            self.energy_gain = food_energy

    def can_reproduce(self) -> bool:
        return self.energy >= self.traits.reproduction_threshold

    def reproduce(
        self,
        world_grid: np.ndarray,
        world_size: int,
        rng: np.random.Generator,
        mutation_rate: float = 0.1,
    ) -> Optional["Creature"]:
        return self._reproduction.reproduce(self, world_grid, world_size, rng, mutation_rate)

    def __hash__(self) -> int:
        return id(self)



from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class MovementBehavior:
    DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    def move(
        self,
        creature,
        target: Optional[Tuple[int, int]],
        world_size: int,
        rng: np.random.Generator,
    ) -> None:
        base_energy_cost = creature.traits.speed * creature.traits.metabolism
        if target is None:
            energy_cost = base_energy_cost * 1.5
            self.random_move(creature, world_size, rng)
        else:
            energy_cost = base_energy_cost
            self.targeted_move(creature, target, world_size)
        energy_cost *= rng.uniform(0.8, 1.2)
        creature.energy -= energy_cost

    def random_move(self, creature, world_size: int, rng: np.random.Generator) -> None:
        speed_penalty = np.square(creature.traits.speed) * 0.1
        creature.energy -= speed_penalty
        direction_idx = rng.integers(0, len(self.DIRECTIONS))
        move = self.DIRECTIONS[direction_idx]
        new_pos = np.array([creature.x, creature.y]) + move
        new_pos = np.clip(new_pos, 0, world_size - 1)
        creature.x, creature.y = int(new_pos[0]), int(new_pos[1])

    def targeted_move(self, creature, target: Tuple[int, int], world_size: int) -> None:
        current_pos = np.array([creature.x, creature.y])
        target_pos = np.array(target)
        distance = int(np.sum(np.abs(target_pos - current_pos)))
        distance_penalty = distance * 0.05 * creature.traits.metabolism
        creature.energy -= distance_penalty
        diff = target_pos - current_pos
        move = np.sign(diff)
        new_pos = current_pos + move
        new_pos = np.clip(new_pos, 0, world_size - 1)
        creature.x, creature.y = int(new_pos[0]), int(new_pos[1])



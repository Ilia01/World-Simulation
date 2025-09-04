from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..entities.traits import Traits


class ReproductionBehavior:
    NEIGHBOR_OFFSETS = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    )

    def mutated_value(
        self,
        value: float,
        mutation_rate: float,
        rng: np.random.Generator,
        min_sd: float = 0.1,
        min_val: Optional[float] = None,
    ) -> float:
        base_sd = np.maximum(mutation_rate * np.abs(value), min_sd)
        mutation = rng.normal(0, base_sd)
        if rng.random() < 0.1:
            mutation += rng.normal(0, base_sd * 3)
        mutated = value + mutation
        return np.maximum(min_val, mutated) if min_val is not None else mutated

    def reproduce(
        self,
        creature,
        world_grid: np.ndarray,
        world_size: int,
        rng: np.random.Generator,
        mutation_rate: float = 0.1,
    ) -> Optional["Creature"]:
        from ..entities.creature import Creature

        if creature.energy < creature.traits.reproduction_threshold:
            return None

        positions = self.NEIGHBOR_OFFSETS + [creature.x, creature.y]
        valid_mask = np.all((positions >= 0) & (positions < world_size), axis=1)
        valid_positions = positions[valid_mask]
        if not len(valid_positions):
            return None

        empty_mask = np.array([world_grid[x, y] is None for x, y in valid_positions])
        empty_positions = valid_positions[empty_mask]
        if not len(empty_positions):
            return None

        chosen_pos = empty_positions[rng.integers(len(empty_positions))]
        creature.energy -= creature.traits.reproduction_cost
        traits = creature.traits
        child_traits = Traits(
            vision=max(
                1,
                int(
                    round(
                        self.mutated_value(traits.vision, mutation_rate, rng, min_sd=0.5, min_val=1)
                    )
                ),
            ),
            speed=max(
                1,
                int(
                    round(
                        self.mutated_value(traits.speed, mutation_rate, rng, min_sd=0.5, min_val=1)
                    )
                ),
            ),
            metabolism=max(
                0.01, self.mutated_value(traits.metabolism, mutation_rate, rng, min_sd=0.01)
            ),
            reproduction_threshold=max(
                1.0,
                self.mutated_value(
                    traits.reproduction_threshold, mutation_rate, rng, min_sd=0.5
                ),
            ),
            reproduction_cost=max(
                0.1, self.mutated_value(traits.reproduction_cost, mutation_rate, rng, min_sd=0.1)
            ),
        )

        return Creature(
            x=int(chosen_pos[0]),
            y=int(chosen_pos[1]),
            energy=creature.traits.reproduction_cost,
            traits=child_traits,
            lineage=creature.lineage,
        )



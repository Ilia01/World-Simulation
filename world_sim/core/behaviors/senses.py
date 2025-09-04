from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class SensingBehavior:
    def find_closest_food(
        self, creature, world_grid: np.ndarray, rng: np.random.Generator
    ) -> Optional[Tuple[int, int]]:
        world_size = world_grid.shape[0]
        vision = creature.traits.vision

        x_min = max(0, creature.x - vision)
        x_max = min(world_size, creature.x + vision + 1)
        y_min = max(0, creature.y - vision)
        y_max = min(world_size, creature.y + vision + 1)

        visible_area = world_grid[x_min:x_max, y_min:y_max]
        food_positions = np.where(visible_area == "F")
        if not food_positions[0].size:
            return None

        rel_x = food_positions[0]
        rel_y = food_positions[1]
        distances = np.abs(rel_x[:, np.newaxis] - 0) + np.abs(rel_y - 0)
        min_dist_idx = np.where(distances == np.min(distances))[0]
        chosen_idx = rng.choice(min_dist_idx)
        target_x = int(rel_x[chosen_idx] + x_min)
        target_y = int(rel_y[chosen_idx] + y_min)
        return (target_x, target_y)



from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np

from ..logging.logger import WorldLogger
from .entities.creature import Creature


class World:
    def __init__(self, size: int = 50, rng_seed: Optional[int] = None, logger: Optional[WorldLogger] = None):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
        self.food_spawn_times: Dict[Tuple[int, int], int] = {}
        self.rng = np.random.default_rng(rng_seed)
        self.logger = logger or WorldLogger()
        self.creatures: List[Creature] = []

        # Enhanced performance metrics
        self.metrics = {
            "start_time": datetime.now(),
            "peak_population": 0,
            "total_food_spawned": 0,
            "births": 0,
            "deaths": 0,
            "total_energy_consumed": 0.0,
            "generations": 0,
            "extinction_events": 0,
            "evolutionary_events": 0,
        }

        # Track histories
        self.population_history: List[int] = []
        self.food_history: List[int] = []

    def spawn_food(self, num: int, tick: int) -> None:
        empty_positions = np.nonzero(self.grid == None)
        empty_coords = list(zip(empty_positions[0], empty_positions[1]))
        if not empty_coords:
            return
        num_to_spawn = min(num, len(empty_coords))
        selected_indices = self.rng.choice(len(empty_coords), size=num_to_spawn, replace=False)
        selected_positions = [empty_coords[i] for i in selected_indices]
        for x, y in selected_positions:
            self.grid[x, y] = "F"
            self.food_spawn_times[(x, y)] = tick
            self.metrics["total_food_spawned"] += 1
            self.logger.log_event("Food spawned", f"({x},{y})", tick)

    def remove_expired_food(self, tick: int, expire_after: int = 20) -> None:
        to_remove = [(x, y) for (x, y), t0 in self.food_spawn_times.items() if (tick - t0) > expire_after]
        if to_remove:
            coords = np.array(to_remove)
            self.grid[coords[:, 0], coords[:, 1]] = None
            for pos in to_remove:
                self.food_spawn_times.pop(pos)
                self.logger.log_event("Food expired", f"({pos[0]},{pos[1]})", tick)

    def add_creature(self, creature: Creature) -> None:
        self.creatures.append(creature)

    def step(
        self,
        tick: int,
        food_spawn_interval: int,
        food_per_spawn: int,
        food_expire: int,
        mutation_rate: float,
    ) -> None:
        if tick % food_spawn_interval == 0:
            self.spawn_food(food_per_spawn, tick)
        if tick % max(1, food_expire) == 0:
            self.remove_expired_food(tick, expire_after=food_expire)

        alive: list[Creature] = []
        new_offspring: list[Creature] = []

        for c in self.creatures:
            if not c.is_alive():
                self.logger.log_event("Creature died", f"Age: {c.age}, Energy: {c.energy:.1f}", tick)
                self.metrics["deaths"] += 1
                self.metrics["total_energy_consumed"] += c.energy
                continue

            c.step(self.grid, self.size, self.rng)

            if c.can_reproduce():
                child = c.reproduce(self.grid, self.size, self.rng, mutation_rate=mutation_rate)
                if child:
                    new_offspring.append(child)
                    self.metrics["births"] += 1
                    self.metrics["evolutionary_events"] += 1
                    self.logger.log_event(
                        "Creature reproduced",
                        f"Parent ID: {self.logger.creature_ids.get(c, 'unknown')}, Energy: {c.energy:.1f}, Position: ({c.x},{c.y})",
                        tick,
                    )
                    # Log offspring snapshot for portfolio analytics
                    try:
                        self.logger.log_creature(child, tick)
                    except Exception:
                        pass

            if c.is_alive():
                alive.append(c)

        for child in new_offspring:
            self.creatures.append(child)
            alive.append(child)
            # Visual birth marker in grid for one tick
            if 0 <= child.x < self.size and 0 <= child.y < self.size:
                # Temporarily mark with 'B' to be picked up by CLI for a birth flash
                if self.grid[child.x, child.y] is None:
                    self.grid[child.x, child.y] = 'B'
        # Clean-up temporary birth markers after rendering window by decaying them quickly
        # Leave them only for this tick – overwrite to None where no food
        birth_mask = (self.grid == 'B')
        if np.any(birth_mask):
            self.grid[birth_mask] = None

        self.creatures = alive

        # Update metrics/history
        current_population = len(self.creatures)
        if current_population > self.metrics["peak_population"]:
            self.metrics["peak_population"] = current_population
        self.population_history.append(current_population)
        self.food_history.append(int(np.sum(self.grid == "F")))
        if current_population == 0:
            self.metrics["extinction_events"] += 1

        food_count = int(np.sum(self.grid == "F"))
        self.logger.record_tick(tick, self.creatures, food_count)
        # Periodic full snapshot for analytics
        if tick % 5 == 0:
            for c in self.creatures:
                self.logger.log_creature(c, tick)



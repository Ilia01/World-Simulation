from __future__ import annotations

import csv
from datetime import datetime
from typing import Dict, List, Optional

from ..core.entities.creature import Creature


class WorldLogger:
    def __init__(self, filename_events: str = "events.csv", filename_creatures: str = "creatures.csv", filename_history: str = "history.csv"):
        self.events_file = open(filename_events, "w", newline="")
        self.creatures_file = open(filename_creatures, "w", newline="")
        self.history_file = open(filename_history, "w", newline="")

        self.events_writer = csv.DictWriter(
            self.events_file, fieldnames=["timestamp", "tick", "event", "details"]
        )
        self.creatures_writer = csv.DictWriter(
            self.creatures_file,
            fieldnames=[
                "tick",
                "creature_id",
                "x",
                "y",
                "energy",
                "age",
                "ate_food",
                "energy_gain",
                "vision",
                "speed",
                "metabolism",
                "lineage",
            ],
        )
        self.history_writer = csv.DictWriter(
            self.history_file,
            fieldnames=["tick", "creature_count", "food_count", "avg_energy", "max_age"],
        )

        self.events_writer.writeheader()
        self.creatures_writer.writeheader()
        self.history_writer.writeheader()
        self.creature_id_counter = 0
        self.creature_ids: Dict[Creature, int] = {}
        self.recent_events: List[Dict] = []
        self.max_recent_events = 10
        self.history: List[Dict] = []

    def log_event(self, event: str, details: str = "", tick: Optional[int] = None) -> None:
        time_str = datetime.now().strftime("%H:%M:%S")
        event_data = {
            "timestamp": time_str,
            "tick": tick if tick is not None else "",
            "event": event,
            "details": details,
        }
        self.recent_events.append(event_data)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)
        self.events_writer.writerow(event_data)
        self.events_file.flush()

    def log_creature(self, c: Creature, tick: Optional[int] = None) -> None:
        if c not in self.creature_ids:
            self.creature_ids[c] = self.creature_id_counter
            self.creature_id_counter += 1
        cid = self.creature_ids[c]
        row = {
            "tick": tick if tick is not None else "",
            "creature_id": cid,
            "x": c.x,
            "y": c.y,
            "energy": c.energy,
            "age": c.age,
            "ate_food": c.ate_food,
            "energy_gain": c.energy_gain,
            "vision": c.traits.vision,
            "speed": c.traits.speed,
            "metabolism": c.traits.metabolism,
            "lineage": c.lineage if c.lineage is not None else "",
        }
        self.creatures_writer.writerow(row)
        self.creatures_file.flush()

    def close(self) -> None:
        self.events_file.close()
        self.creatures_file.close()
        self.history_file.close()

    def record_tick(self, tick: int, creatures: List[Creature], food_count: int) -> None:
        if creatures:
            energies = [c.energy for c in creatures]
            ages = [c.age for c in creatures]
            avg_energy = float(sum(energies) / len(energies))
            max_age = int(max(ages))
        else:
            avg_energy = 0.0
            max_age = 0
        row = {
            "tick": tick,
            "creature_count": len(creatures),
            "food_count": int(food_count),
            "avg_energy": avg_energy,
            "max_age": max_age,
        }
        self.history.append(row)
        self.history_writer.writerow(row)
        self.history_file.flush()

    def get_recent_events(self, num: int = 5) -> List[Dict]:
        return self.recent_events[-num:]



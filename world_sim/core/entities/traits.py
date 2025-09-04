from dataclasses import dataclass


@dataclass
class Traits:
    vision: int = 3
    speed: int = 1
    metabolism: float = 0.2
    reproduction_threshold: float = 20.0
    reproduction_cost: float = 8.0
    max_age: int = 200

    def to_dict(self) -> dict:
        return {
            "vision": self.vision,
            "speed": self.speed,
            "metabolism": self.metabolism,
            "reproduction_threshold": self.reproduction_threshold,
            "reproduction_cost": self.reproduction_cost,
            "max_age": self.max_age,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Traits":
        return cls(
            vision=int(data.get("vision", 3)),
            speed=int(data.get("speed", 1)),
            metabolism=float(data.get("metabolism", 0.2)),
            reproduction_threshold=float(data.get("reproduction_threshold", 20.0)),
            reproduction_cost=float(data.get("reproduction_cost", 8.0)),
            max_age=int(data.get("max_age", 200)),
        )



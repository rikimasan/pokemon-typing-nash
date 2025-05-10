from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True, frozen=True)
class Pokemon:
    types: Tuple[str, ...]
    level: int = 50

    base_hp: int = 90
    base_atk: int = 90
    base_def: int = 90
    base_spa: int = 90
    base_spd: int = 90
    base_spe: int = 90

    def __post_init__(self) -> None:
        if not 1 <= len(self.types) <= 2:
            raise ValueError("Pokemon must be mono- or dual-typed.")

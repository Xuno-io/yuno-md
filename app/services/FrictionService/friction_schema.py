"""
Friction schema: types for the FrictionEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FrictionLevel(IntEnum):
    VALIDACION = 0  # Solo saludos / bajo esfuerzo
    ANALISIS = 1  # Raro - reservado para uso futuro
    DESAFIO = 2  # Default para cualquier mensaje concreto
    ANIQUILACION = 3  # Usuario repite el mismo punto


@dataclass
class FrictionConstraints:
    required_min: FrictionLevel
    is_repeating: bool
    is_vague: bool

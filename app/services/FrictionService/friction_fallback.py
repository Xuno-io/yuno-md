"""
FrictionFallbackGenerator: genera fricción determinista cuando el LLM no llama
la herramienta emit_friction_response o viola las restricciones de nivel.
"""

from __future__ import annotations

import random

from app.services.FrictionService.friction_schema import FrictionConstraints


class FrictionFallbackGenerator:
    """Genera respuestas de fricción sin pasar por el LLM."""

    TEMPLATES: dict[str, list[str]] = {
        "repeticion": [
            "Llevas varias iteraciones planteando variaciones del mismo punto. "
            "El problema no está en la formulación — está en que evitas comprometerte "
            "con una respuesta. ¿Cuál es el obstáculo concreto detrás de '{core}'?",
            "La repetición es una señal: algo en '{core}' genera resistencia. "
            "Nombra el obstáculo específico o cambia el ángulo de ataque.",
        ],
        "vaguedad": [
            "'{core}' no es suficiente para trabajar con ello. "
            "Dame el problema específico: qué tienes, qué quieres, qué te lo impide.",
            "Eso es un impulso, no una pregunta. "
            "Reformúlalo con contexto concreto: situación actual, objetivo, bloqueo.",
        ],
        "desobediencia": [
            "Protocolo de respuesta no ejecutado. Reformula la consulta con más precisión.",
            "Respuesta inválida. Presenta el argumento con más rigor para continuar.",
        ],
    }

    def generate(
        self,
        user_message: str,
        constraints: FrictionConstraints,
        violation_reason: str,
    ) -> str:
        """
        Genera una respuesta de fricción basada en el tipo de violación.

        Args:
            user_message: Mensaje original del usuario.
            constraints: Restricciones calculadas por FrictionEngine.
            violation_reason: "tool_not_called", "repeticion", "vaguedad".

        Returns:
            String de fricción para enviar al usuario.
        """
        core = user_message[:100]

        if violation_reason == "tool_not_called":
            if constraints.is_repeating:
                templates = self.TEMPLATES["repeticion"]
            elif constraints.is_vague:
                templates = self.TEMPLATES["vaguedad"]
            else:
                templates = self.TEMPLATES["desobediencia"]
        elif violation_reason == "repeticion":
            templates = self.TEMPLATES["repeticion"]
        elif violation_reason == "vaguedad":
            templates = self.TEMPLATES["vaguedad"]
        else:
            templates = self.TEMPLATES["desobediencia"]

        template = random.choice(templates)  # nosec B311 — non-security template selector
        return template.format(core=core)

"""
FrictionEngine: detección determinista del nivel de fricción requerido.
"""

from __future__ import annotations

import re

from app.services.FrictionService.friction_schema import (
    FrictionConstraints,
    FrictionLevel,
)

# Strips "[Group: ...][User: ...]: " prefixes added by TelegramService for group chats
_GROUP_PREFIX_RE = re.compile(r"^\[Group:[^\]]+\]\[User:[^\]]+\]:\s*")

GREETING_WORDS = {
    "hola",
    "hi",
    "hello",
    "hey",
    "buenas",
    "buenos",
    "buen",
    "saludos",
    "ola",
    "ey",
    "que tal",
    "qué tal",
}

VAGUE_PHRASES = {
    "ayuda",
    "no sé",
    "no se",
    "cómo",
    "como",
    "qué hago",
    "que hago",
    "help",
    "no entiendo",
    "no lo entiendo",
}


class FrictionEngine:
    """Calcula el nivel mínimo de fricción requerido para un mensaje de usuario."""

    def _strip_group_prefix(self, msg: str) -> str:
        """Remove '[Group: ...][User: ...]: ' prefix added by TelegramService."""
        return _GROUP_PREFIX_RE.sub("", msg).strip()

    def calculate_required_friction(
        self,
        user_message: str,
        history: list,
    ) -> FrictionConstraints:
        """
        Analiza el mensaje del usuario y el historial para determinar
        el nivel de fricción mínimo requerido.

        Args:
            user_message: Último mensaje del usuario (puede incluir prefijo de grupo).
            history: Historial previo (sin incluir el último mensaje).

        Returns:
            FrictionConstraints con el nivel mínimo y flags descriptivos.
        """
        # Strip group prefix before running any detection
        msg = self._strip_group_prefix(user_message)

        if self._is_greeting(msg):
            return FrictionConstraints(
                required_min=FrictionLevel.VALIDACION,
                is_repeating=False,
                is_vague=False,
            )

        is_repeating = self._detect_repetition(msg, history)
        if is_repeating:
            return FrictionConstraints(
                required_min=FrictionLevel.ANIQUILACION,
                is_repeating=True,
                is_vague=False,
            )

        is_vague = self._is_vague(msg)
        return FrictionConstraints(
            required_min=FrictionLevel.DESAFIO,
            is_repeating=False,
            is_vague=is_vague,
        )

    def _is_greeting(self, msg: str) -> bool:
        """True si el mensaje es un saludo de bajo esfuerzo (< 5 palabras)."""
        import string

        words = msg.strip().lower().split()
        if len(words) >= 5:
            return False
        if not words:
            return False
        normalized = [
            w.strip(string.punctuation) for w in words if w.strip(string.punctuation)
        ]
        if not normalized:
            return False
        # Check two-word greetings first ("que tal", "qué tal")
        if len(normalized) >= 2 and " ".join(normalized[:2]) in GREETING_WORDS:
            return True
        return normalized[0] in GREETING_WORDS

    def _detect_repetition(self, msg: str, history: list) -> bool:
        """
        True si el usuario está repitiendo el mismo punto con más del 60%
        de similitud de palabras en al menos 2 mensajes previos (últimos 5).
        """
        user_messages = [
            self._strip_group_prefix(m.get("content", ""))
            for m in history[-5:]
            if m.get("role") == "user"
        ]
        if not user_messages:
            return False

        msg_words = set(msg.lower().split())
        if not msg_words:
            return False

        similar_count = 0
        for prev_msg in user_messages:
            prev_words = set(prev_msg.lower().split())
            if not prev_words:
                continue
            union = msg_words | prev_words
            intersection = msg_words & prev_words
            similarity = len(intersection) / len(union)
            if similarity >= 0.6:
                similar_count += 1

        return similar_count >= 2

    def _is_vague(self, msg: str) -> bool:
        """True si el mensaje es demasiado corto o contiene frases vagas."""
        words = msg.strip().split()
        if len(words) < 5:
            return True
        msg_lower = msg.lower()
        return any(phrase in msg_lower for phrase in VAGUE_PHRASES)

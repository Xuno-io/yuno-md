"""
Unit tests for FrictionEngine.
"""

import pytest

from app.services.FrictionService.friction_engine import FrictionEngine
from app.services.FrictionService.friction_schema import FrictionLevel
from app.entities.message import MessagePayload


@pytest.fixture
def engine() -> FrictionEngine:
    return FrictionEngine()


class TestIsGreeting:
    """Tests for greeting detection."""

    def test_single_hola(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("hola", [])
        assert result.required_min == FrictionLevel.VALIDACION

    def test_greeting_with_exclamation(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("hola!", [])
        assert result.required_min == FrictionLevel.VALIDACION

    def test_greeting_hi(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("hi", [])
        assert result.required_min == FrictionLevel.VALIDACION

    def test_greeting_hello(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("hello", [])
        assert result.required_min == FrictionLevel.VALIDACION

    def test_greeting_buenas(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("buenas tardes", [])
        assert result.required_min == FrictionLevel.VALIDACION

    def test_long_greeting_is_not_greeting(self, engine: FrictionEngine) -> None:
        # 5+ words — should not be treated as a greeting
        result = engine.calculate_required_friction(
            "hola quiero preguntarte sobre algo importante", []
        )
        assert result.required_min != FrictionLevel.VALIDACION

    def test_non_greeting_word_start(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("tengo un problema", [])
        assert result.required_min != FrictionLevel.VALIDACION

    def test_greeting_flags(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("hola", [])
        assert result.is_repeating is False
        assert result.is_vague is False


class TestDetectRepetition:
    """Tests for repetition detection."""

    def test_no_history_no_repetition(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("cuéntame algo", [])
        assert result.is_repeating is False

    def test_similar_messages_trigger_aniquilacion(
        self, engine: FrictionEngine
    ) -> None:
        # Both user messages have >=0.6 similarity with the current message
        history: list[MessagePayload] = [
            {"role": "user", "content": "mi startup necesita financiación urgente", "attachments": []},
            {"role": "assistant", "content": "respuesta", "attachments": []},
            {"role": "user", "content": "mi startup necesita financiación urgente ya", "attachments": []},
            {"role": "assistant", "content": "respuesta", "attachments": []},
        ]
        result = engine.calculate_required_friction(
            "mi startup necesita financiación urgente", history
        )
        assert result.required_min == FrictionLevel.ANIQUILACION
        assert result.is_repeating is True

    def test_different_messages_no_repetition(self, engine: FrictionEngine) -> None:
        history: list[MessagePayload] = [
            {"role": "user", "content": "hola qué tal estás hoy", "attachments": []},
            {"role": "assistant", "content": "bien", "attachments": []},
            {"role": "user", "content": "cuéntame sobre python y sus ventajas", "attachments": []},
        ]
        result = engine.calculate_required_friction(
            "prefiero usar rust para este proyecto concreto", history
        )
        assert result.is_repeating is False

    def test_only_one_similar_message_no_repetition(
        self, engine: FrictionEngine
    ) -> None:
        history: list[MessagePayload] = [
            {"role": "user", "content": "necesito ayuda con mi startup", "attachments": []},
            {"role": "assistant", "content": "respuesta", "attachments": []},
            {"role": "user", "content": "cuéntame sobre ventas", "attachments": []},
        ]
        result = engine.calculate_required_friction(
            "necesito ayuda con mi startup", history
        )
        # Only 1 similar message (not 2) → no repetition
        assert result.is_repeating is False

    def test_only_user_messages_count(self, engine: FrictionEngine) -> None:
        # Assistant messages should NOT count toward repetition
        history: list[MessagePayload] = [
            {"role": "assistant", "content": "necesito ayuda con mi startup", "attachments": []},
            {"role": "assistant", "content": "necesito ayuda con mi startup", "attachments": []},
            {"role": "assistant", "content": "necesito ayuda con mi startup", "attachments": []},
        ]
        result = engine.calculate_required_friction(
            "necesito ayuda con mi startup", history
        )
        assert result.is_repeating is False


class TestIsVague:
    """Tests for vagueness detection."""

    def test_short_message_is_vague(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("ayuda por favor", [])
        assert result.is_vague is True

    def test_vague_phrase_ayuda(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "necesito ayuda con esto pero no sé bien qué", []
        )
        assert result.is_vague is True

    def test_vague_phrase_no_se(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "no sé qué hacer con mi problema aquí", []
        )
        assert result.is_vague is True

    def test_concrete_message_not_vague(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "estoy migrando de PostgreSQL a DynamoDB y tengo dudas sobre el modelo de datos", []
        )
        assert result.is_vague is False


class TestGroupPrefixStripping:
    """Tests for group chat prefix removal before detection."""

    def test_group_greeting_detected(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "[Group: pruebas][User: jackcloudmann]: hola", []
        )
        assert result.required_min == FrictionLevel.VALIDACION

    def test_group_vague_message_detected(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "[Group: pruebas][User: jackcloudmann]: ayuda", []
        )
        assert result.is_vague is True

    def test_group_concrete_message_is_desafio(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "[Group: equipo][User: alice]: estoy migrando de PostgreSQL a DynamoDB y necesito entender los trade-offs",
            [],
        )
        assert result.required_min == FrictionLevel.DESAFIO
        assert result.is_vague is False


class TestDefaultDesafio:
    """Tests for the DESAFIO default level."""

    def test_concrete_message_is_desafio(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction(
            "estoy construyendo una API REST con FastAPI y necesito entender los trade-offs", []
        )
        assert result.required_min == FrictionLevel.DESAFIO
        assert result.is_repeating is False

    def test_vague_message_is_still_desafio(self, engine: FrictionEngine) -> None:
        result = engine.calculate_required_friction("ayuda", [])
        assert result.required_min == FrictionLevel.DESAFIO
        assert result.is_vague is True

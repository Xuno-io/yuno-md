from abc import ABC, abstractmethod

from app.entities.message import MessagePayload

from app.services.MemoryService.memory_service_interface import (
    MemoryServiceInterface,
)


class NeibotServiceInterface(ABC):
    # Optional memory service - implementations may or may not have this
    memory_service: MemoryServiceInterface | None = None

    @abstractmethod
    async def get_response(
        self,
        history: list[MessagePayload],
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Return the assistant reply for the provided conversation history."""

    @abstractmethod
    async def capture_facts_from_history(
        self,
        history: list[MessagePayload],
        user_id: str,
    ) -> int:
        """
        Analyze history, extract facts, and save them to the knowledge base.
        Returns the number of facts saved.
        """

    @abstractmethod
    async def distill_response(
        self,
        original_response: str,
        context: list[MessagePayload] | None = None,
    ) -> str:
        """
        Distill a long response into a condensed version using the fragmentation protocol.

        This method is called when a response exceeds Telegram's character limit.
        It applies the 4-movement distillation protocol to condense the response.

        Args:
            original_response: The original response that was too long
            context: Optional recent conversation history (last 10 messages) for better context

        Returns:
            A condensed version of the response following the distillation protocol
        """

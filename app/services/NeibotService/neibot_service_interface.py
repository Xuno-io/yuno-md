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

from abc import ABC, abstractmethod

from app.entities.message import MessagePayload


class NeibotServiceInterface(ABC):
    @abstractmethod
    async def get_response(self, history: list[MessagePayload]) -> str:
        """Return the assistant reply for the provided conversation history."""

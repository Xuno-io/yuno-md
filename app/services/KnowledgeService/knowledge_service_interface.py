from abc import ABC, abstractmethod
from typing import Any


class KnowledgeServiceInterface(ABC):
    """Interface for interacting with the Knowledge Core service."""

    @abstractmethod
    def health(self) -> bool:
        """Check if the knowledge service is healthy."""
        pass

    @abstractmethod
    def add_fact(self, entity: str, attribute: str, value: Any) -> bool:
        """Add a fact to the knowledge base."""
        pass

    @abstractmethod
    def query(self, pattern: dict[str, Any]) -> list[dict[str, Any]]:
        """Query facts that match the provided pattern."""
        pass

    @abstractmethod
    def invalidate(self, fact_id: str) -> bool:
        """Invalidate (delete) a fact by id."""
        pass

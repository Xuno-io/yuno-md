from abc import ABC, abstractmethod
from typing import Any


class MemoryServiceInterface(ABC):
    """Interface for the long-term memory service using mem0."""

    @abstractmethod
    def add(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a memory to the store.

        Args:
            content: The text content to memorize
            user_id: The user identifier
            metadata: Optional metadata (category, confidence, etc.)

        Returns:
            Result from mem0 including memory IDs created
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search memories by semantic similarity with optional category filter.

        Args:
            query: The search query
            user_id: The user identifier
            category: Optional category filter (TECH_STACK, BUSINESS_LOGIC, USER_CONSTRAINTS)
            limit: Maximum number of results

        Returns:
            List of matching memories with scores
        """
        pass

    @abstractmethod
    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get all memories for a user.

        Args:
            user_id: The user identifier

        Returns:
            List of all memories for this user
        """
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The memory identifier

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    def delete_all(self, user_id: str) -> int:
        """
        Delete all memories for a user.

        Args:
            user_id: The user identifier

        Returns:
            Number of memories deleted
        """
        pass

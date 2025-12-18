"""
Unit tests for MemoryService using mem0.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.services.MemoryService.memory_service import (
    MemoryService,
    parse_embedded_category,
    VALID_CATEGORIES,
)


class TestParseEmbeddedCategory:
    """Tests for the parse_embedded_category helper function."""

    def test_parse_valid_tech_stack_category(self) -> None:
        """Test parsing a valid TECH_STACK category."""
        text = "[TECH_STACK] Uses Redis for caching"
        category, clean_text = parse_embedded_category(text)

        assert category == "TECH_STACK"
        assert clean_text == "Uses Redis for caching"

    def test_parse_valid_business_logic_category(self) -> None:
        """Test parsing a valid BUSINESS_LOGIC category."""
        text = "[BUSINESS_LOGIC] Budget limit is 500 USD"
        category, clean_text = parse_embedded_category(text)

        assert category == "BUSINESS_LOGIC"
        assert clean_text == "Budget limit is 500 USD"

    def test_parse_valid_user_constraints_category(self) -> None:
        """Test parsing a valid USER_CONSTRAINTS category."""
        text = "[USER_CONSTRAINTS] Prefers async over sync"
        category, clean_text = parse_embedded_category(text)

        assert category == "USER_CONSTRAINTS"
        assert clean_text == "Prefers async over sync"

    def test_parse_unknown_category_returns_other(self) -> None:
        """Test that unknown categories default to OTHER."""
        text = "[UNKNOWN_CAT] Some text here"
        category, clean_text = parse_embedded_category(text)

        assert category == "OTHER"
        assert clean_text == "[UNKNOWN_CAT] Some text here"

    def test_parse_no_category_returns_other(self) -> None:
        """Test that text without category returns OTHER."""
        text = "Just plain text without category"
        category, clean_text = parse_embedded_category(text)

        assert category == "OTHER"
        assert clean_text == "Just plain text without category"

    def test_parse_malformed_bracket_returns_other(self) -> None:
        """Test that malformed brackets return OTHER."""
        text = "[TECH_STACK Some text"  # Missing closing bracket
        category, clean_text = parse_embedded_category(text)

        assert category == "OTHER"
        assert clean_text == "[TECH_STACK Some text"

    def test_valid_categories_constant(self) -> None:
        """Test that VALID_CATEGORIES contains expected values."""
        assert "TECH_STACK" in VALID_CATEGORIES
        assert "BUSINESS_LOGIC" in VALID_CATEGORIES
        assert "USER_CONSTRAINTS" in VALID_CATEGORIES
        assert len(VALID_CATEGORIES) == 3


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    return logging.getLogger("MemoryServiceTest")


@pytest.fixture
def mock_memory():
    """Create a mock mem0 Memory instance."""
    return MagicMock()


@pytest.fixture
def memory_service(logger: logging.Logger, mock_memory: MagicMock) -> MemoryService:
    """Create a MemoryService with mocked mem0 backend."""
    with patch(
        "app.services.MemoryService.memory_service.Memory.from_config"
    ) as mock_from_config:
        mock_from_config.return_value = mock_memory
        service = MemoryService(
            logger=logger,
            redis_host="localhost",
            redis_port=6379,
        )
        return service


class TestMemoryServiceAdd:
    """Tests for MemoryService.add method."""

    def test_add_memory_success(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test successful memory addition."""
        mock_memory.add.return_value = {"results": [{"id": "mem_123"}]}

        result = memory_service.add(
            content="User prefers Python",
            user_id="user_456",
        )

        mock_memory.add.assert_called_once()
        assert "results" in result

    def test_add_memory_with_metadata(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test memory addition with metadata."""
        mock_memory.add.return_value = {"results": []}

        memory_service.add(
            content="Uses FastAPI",
            user_id="user_789",
            metadata={"source": "conversation"},
        )

        call_args = mock_memory.add.call_args
        assert call_args.kwargs["metadata"] == {"source": "conversation"}

    def test_add_memory_handles_exception(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test that exceptions are caught and returned as error dict."""
        mock_memory.add.side_effect = Exception("Connection failed")

        result = memory_service.add(
            content="Test content",
            user_id="user_123",
        )

        assert "error" in result
        assert "Connection failed" in result["error"]


class TestMemoryServiceSearch:
    """Tests for MemoryService.search method."""

    def test_search_returns_results(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test successful search."""
        mock_memory.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "[TECH_STACK] Uses Python"},
                {"id": "mem_2", "memory": "[TECH_STACK] Uses Redis"},
            ]
        }

        results = memory_service.search(
            query="What tech stack?",
            user_id="user_123",
            limit=5,
        )

        assert len(results) == 2
        mock_memory.search.assert_called_once()

    def test_search_with_category_filter(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test search with category filtering."""
        mock_memory.search.return_value = {
            "results": [
                {"id": "mem_1", "memory": "[TECH_STACK] Uses Python"},
                {"id": "mem_2", "memory": "[BUSINESS_LOGIC] Budget is 500"},
            ]
        }

        results = memory_service.search(
            query="What do you know?",
            user_id="user_123",
            category="TECH_STACK",
            limit=5,
        )

        # Should filter to only TECH_STACK
        assert len(results) == 1
        assert "[TECH_STACK]" in results[0]["memory"]

    def test_search_empty_results(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test search with no results."""
        mock_memory.search.return_value = {"results": []}

        results = memory_service.search(
            query="Something unknown",
            user_id="user_123",
        )

        assert results == []

    def test_search_handles_exception(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test that exceptions return empty list."""
        mock_memory.search.side_effect = Exception("Search failed")

        results = memory_service.search(
            query="Test query",
            user_id="user_123",
        )

        assert results == []


class TestMemoryServiceGetAll:
    """Tests for MemoryService.get_all method."""

    def test_get_all_returns_memories(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test getting all memories for a user."""
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem_1", "memory": "Fact 1"},
                {"id": "mem_2", "memory": "Fact 2"},
            ]
        }

        results = memory_service.get_all(user_id="user_123")

        assert len(results) == 2
        mock_memory.get_all.assert_called_once_with(user_id="user_123")

    def test_get_all_empty(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test getting all memories when none exist."""
        mock_memory.get_all.return_value = {"results": []}

        results = memory_service.get_all(user_id="user_123")

        assert results == []

    def test_get_all_handles_exception(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test that exceptions return empty list."""
        mock_memory.get_all.side_effect = Exception("Failed")

        results = memory_service.get_all(user_id="user_123")

        assert results == []


class TestMemoryServiceDelete:
    """Tests for MemoryService.delete method."""

    def test_delete_success(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test successful memory deletion."""
        mock_memory.delete.return_value = None

        result = memory_service.delete(memory_id="mem_123")

        assert result is True
        mock_memory.delete.assert_called_once_with("mem_123")

    def test_delete_handles_exception(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test that exceptions return False."""
        mock_memory.delete.side_effect = Exception("Delete failed")

        result = memory_service.delete(memory_id="mem_123")

        assert result is False


class TestMemoryServiceDeleteAll:
    """Tests for MemoryService.delete_all method."""

    def test_delete_all_success(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test deleting all memories for a user."""
        mock_memory.get_all.return_value = {
            "results": [
                {"id": "mem_1"},
                {"id": "mem_2"},
            ]
        }
        mock_memory.delete.return_value = None

        count = memory_service.delete_all(user_id="user_123")

        assert count == 2
        assert mock_memory.delete.call_count == 2

    def test_delete_all_empty(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test delete_all when no memories exist."""
        mock_memory.get_all.return_value = {"results": []}

        count = memory_service.delete_all(user_id="user_123")

        assert count == 0

    def test_delete_all_handles_exception(
        self, memory_service: MemoryService, mock_memory: MagicMock
    ) -> None:
        """Test that exceptions return 0."""
        mock_memory.get_all.side_effect = Exception("Failed")

        count = memory_service.delete_all(user_id="user_123")

        assert count == 0

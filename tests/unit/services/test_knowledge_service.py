"""
Unit tests for KnowledgeService.

Tests cover the public API: health, add_fact, query, invalidate.
Uses a mock HTTP client to avoid network dependencies.
"""

import logging
from typing import Any

import pytest

from app.services.KnowledgeService.knowledge_service import KnowledgeService


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(
        self,
        json_data: dict[str, Any],
        status_code: int = 200,
        raise_on_status: bool = False,
    ):
        self._json_data = json_data
        self.status_code = status_code
        self._raise_on_status = raise_on_status

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self._raise_on_status:
            raise Exception(f"HTTP {self.status_code}")


class MockHttpClient:
    """Mock HTTP client that implements the HttpClientProtocol."""

    def __init__(self):
        self.get_response: MockResponse | None = None
        self.post_response: MockResponse | None = None
        self.get_exception: Exception | None = None
        self.post_exception: Exception | None = None
        self.call_count = {"get": 0, "post": 0}
        self.last_post_payload: dict[str, Any] | None = None
        self.last_post_url: str | None = None

    def get(self, url: str, *, timeout: Any = None, **kwargs) -> MockResponse:
        self.call_count["get"] += 1
        if self.get_exception:
            raise self.get_exception
        if self.get_response is None:
            raise Exception("No mock response configured for GET")
        return self.get_response

    def post(
        self,
        url: str,
        *,
        json: Any = None,
        timeout: Any = None,
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> MockResponse:
        self.call_count["post"] += 1
        self.last_post_url = url
        self.last_post_payload = json
        if self.post_exception:
            raise self.post_exception
        if self.post_response is None:
            raise Exception("No mock response configured for POST")
        return self.post_response


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    return logging.getLogger("KnowledgeServiceTest")


@pytest.fixture
def mock_http() -> MockHttpClient:
    """Create a mock HTTP client."""
    return MockHttpClient()


@pytest.fixture
def service(logger: logging.Logger, mock_http: MockHttpClient) -> KnowledgeService:
    """Create a KnowledgeService with mocked HTTP client."""
    return KnowledgeService(
        base_url="http://test-server:8088",
        logger=logger,
        timeout=1.0,
        http_client=mock_http,
        health_max_attempts=2,
        health_base_backoff=0.01,
        health_overall_timeout=0.5,
    )


class TestHealth:
    """Test cases for health() method."""

    def test_health_success_with_ok_true(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test health returns True when server responds with ok: true."""
        mock_http.get_response = MockResponse({"ok": True})

        result = service.health()

        assert result is True
        assert mock_http.call_count["get"] == 1

    def test_health_success_with_status_ok(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test health returns True when server responds with status: ok."""
        mock_http.get_response = MockResponse({"status": "ok"})

        result = service.health()

        assert result is True

    def test_health_failure_on_network_error(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test health returns False on network error after retries."""
        mock_http.get_exception = Exception("Connection refused")

        result = service.health()

        assert result is False
        # Should have retried (health_max_attempts=2)
        assert mock_http.call_count["get"] == 2

    def test_health_failure_on_unexpected_body(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test health returns False when response body is unexpected."""
        mock_http.get_response = MockResponse({"unexpected": "data"})

        result = service.health()

        assert result is False

    def test_health_failure_on_http_error(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test health returns False on HTTP error status."""
        mock_http.get_response = MockResponse(
            {"error": "Internal Server Error"},
            status_code=500,
            raise_on_status=True,
        )

        result = service.health()

        assert result is False


class TestAddFact:
    """Test cases for add_fact() method."""

    def test_add_fact_success(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test add_fact returns True on successful creation."""
        mock_http.post_response = MockResponse({"ok": True, "id": "f-123"})

        result = service.add_fact(
            entity="user:123",
            attribute="name",
            value="John",
        )

        assert result is True
        assert mock_http.last_post_url == "http://test-server:8088/add-fact"
        assert mock_http.last_post_payload == {
            "entity": "user:123",
            "attribute": "name",
            "value": "John",
        }

    def test_add_fact_failure_ok_false(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test add_fact returns False when server responds ok: false."""
        mock_http.post_response = MockResponse(
            {"ok": False, "error": "Validation error"}
        )

        result = service.add_fact(
            entity="user:123",
            attribute="age",
            value="not-an-integer",
        )

        assert result is False

    def test_add_fact_failure_on_network_error(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test add_fact returns False on network error."""
        mock_http.post_exception = Exception("Connection timeout")

        result = service.add_fact(
            entity="user:123",
            attribute="name",
            value="John",
        )

        assert result is False

    def test_add_fact_with_complex_value(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test add_fact works with complex JSON values."""
        mock_http.post_response = MockResponse({"ok": True, "id": "f-456"})

        result = service.add_fact(
            entity="user:123",
            attribute="preferences",
            value={"theme": "dark", "notifications": True},
        )

        assert result is True
        assert mock_http.last_post_payload is not None
        assert mock_http.last_post_payload["value"] == {
            "theme": "dark",
            "notifications": True,
        }


class TestQuery:
    """Test cases for query() method."""

    def test_query_returns_facts(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test query returns list of facts."""
        mock_http.post_response = MockResponse(
            {
                "facts": [
                    {
                        "id": "f-1",
                        "entity": "user:123",
                        "attribute": "name",
                        "value": "John",
                    },
                    {
                        "id": "f-2",
                        "entity": "user:123",
                        "attribute": "age",
                        "value": 30,
                    },
                ]
            }
        )

        result = service.query({"entity": "user:123"})

        assert len(result) == 2
        assert result[0]["id"] == "f-1"
        assert result[1]["attribute"] == "age"
        assert mock_http.last_post_url == "http://test-server:8088/query"

    def test_query_returns_empty_list_on_no_matches(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test query returns empty list when no facts match."""
        mock_http.post_response = MockResponse({"facts": []})

        result = service.query({"entity": "nonexistent"})

        assert result == []

    def test_query_returns_empty_list_on_error(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test query returns empty list on network error."""
        mock_http.post_exception = Exception("Network error")

        result = service.query({"entity": "user:123"})

        assert result == []

    def test_query_with_multiple_filters(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test query with multiple pattern filters."""
        mock_http.post_response = MockResponse(
            {
                "facts": [
                    {
                        "id": "f-1",
                        "entity": "user:123",
                        "attribute": "name",
                        "value": "John",
                    }
                ]
            }
        )

        result = service.query(
            {
                "entity": "user:123",
                "attribute": "name",
            }
        )

        assert len(result) == 1
        # Verify the pattern was sent correctly (wrapped in 'pattern' key)
        assert mock_http.last_post_payload == {
            "pattern": {"entity": "user:123", "attribute": "name"}
        }


class TestInvalidate:
    """Test cases for invalidate() method."""

    def test_invalidate_success(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test invalidate returns True on success."""
        mock_http.post_response = MockResponse({"ok": True})

        result = service.invalidate("f-123")

        assert result is True
        assert mock_http.last_post_url == "http://test-server:8088/invalidate-fact"
        assert mock_http.last_post_payload == {"id": "f-123"}

    def test_invalidate_failure_fact_not_found(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test invalidate returns False when fact not found."""
        mock_http.post_response = MockResponse({"ok": False})

        result = service.invalidate("nonexistent-id")

        assert result is False

    def test_invalidate_failure_on_network_error(
        self, service: KnowledgeService, mock_http: MockHttpClient
    ) -> None:
        """Test invalidate returns False on network error."""
        mock_http.post_exception = Exception("Connection refused")

        result = service.invalidate("f-123")

        assert result is False


class TestInitialization:
    """Test cases for service initialization."""

    def test_default_base_url_from_env(
        self,
        logger: logging.Logger,
        mock_http: MockHttpClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test base_url is read from environment variable."""
        monkeypatch.setenv("KNOWLEDGE_BASE_URL", "http://env-server:9000")

        service = KnowledgeService(
            logger=logger,
            http_client=mock_http,
        )

        assert service.base_url == "http://env-server:9000"

    def test_explicit_base_url_overrides_env(
        self,
        logger: logging.Logger,
        mock_http: MockHttpClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test explicit base_url overrides environment variable."""
        monkeypatch.setenv("KNOWLEDGE_BASE_URL", "http://env-server:9000")

        service = KnowledgeService(
            base_url="http://explicit:8080",
            logger=logger,
            http_client=mock_http,
        )

        assert service.base_url == "http://explicit:8080"

    def test_trailing_slash_is_stripped(
        self, logger: logging.Logger, mock_http: MockHttpClient
    ) -> None:
        """Test trailing slash is stripped from base_url."""
        service = KnowledgeService(
            base_url="http://server:8088/",
            logger=logger,
            http_client=mock_http,
        )

        assert service.base_url == "http://server:8088"

"""
Unit tests for the web_search_tool module.

Tests cover the web search functionality using Gemini Flash with Google Search grounding.
"""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.tools.web_search_tool import web_search, _get_client


class TestGetClient:
    """Test cases for client initialization."""

    def test_get_client_creates_vertexai_client(self) -> None:
        """Test that client is created with Vertex AI configuration."""
        with patch("app.tools.web_search_tool.genai.Client") as MockClient:
            with patch("app.tools.web_search_tool.os.getenv") as mock_getenv:
                mock_getenv.side_effect = lambda key, default="": {
                    "VERTEX_PROJECT_ID": "test-project",
                    "VERTEX_LOCATION": "us-central1",
                }.get(key, default)

                # Reset the module-level client
                import app.tools.web_search_tool as tool_module

                tool_module._client = None

                _get_client()

                MockClient.assert_called_once_with(
                    vertexai=True,
                    project="test-project",
                    location="us-central1",
                )

    def test_get_client_caches_instance(self) -> None:
        """Test that client instance is cached."""
        with patch("app.tools.web_search_tool.genai.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance

            import app.tools.web_search_tool as tool_module

            tool_module._client = None

            client1 = _get_client()
            client2 = _get_client()

            # Should only create once
            assert MockClient.call_count == 1
            assert client1 is client2


class TestWebSearch:
    """Test cases for the web_search function."""

    @pytest.mark.asyncio
    async def test_web_search_empty_query(self) -> None:
        """Test that empty query returns error."""
        result = await web_search("")

        assert result["status"] == "error"
        assert result["query"] == ""
        assert "Empty query" in result["result"]
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_web_search_whitespace_query(self) -> None:
        """Test that whitespace-only query returns error."""
        result = await web_search("   ")

        assert result["status"] == "error"
        assert "Empty query" in result["result"]

    @pytest.mark.asyncio
    async def test_web_search_success(self) -> None:
        """Test successful web search."""
        mock_response = MagicMock()
        mock_response.text = "Search result summary"
        mock_response.candidates = []

        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            mock_get_client.return_value = mock_client

            result = await web_search("What is Python?")

        assert result["status"] == "success"
        assert result["query"] == "What is Python?"
        assert result["result"] == "Search result summary"
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_web_search_with_grounding_metadata(self) -> None:
        """Test web search extracts grounding sources."""
        mock_web_chunk = MagicMock()
        mock_web_chunk.title = "Python.org"
        mock_web_chunk.uri = "https://python.org"

        mock_chunk = MagicMock()
        mock_chunk.web = mock_web_chunk

        mock_grounding = MagicMock()
        mock_grounding.grounding_chunks = [mock_chunk]

        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = mock_grounding

        mock_response = MagicMock()
        mock_response.text = "Python is a programming language"
        mock_response.candidates = [mock_candidate]

        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            mock_get_client.return_value = mock_client

            result = await web_search("What is Python?")

        assert result["status"] == "success"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Python.org"
        assert result["sources"][0]["uri"] == "https://python.org"

    @pytest.mark.asyncio
    async def test_web_search_empty_response(self) -> None:
        """Test handling of empty response from API."""
        mock_response = MagicMock()
        mock_response.text = ""

        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            mock_get_client.return_value = mock_client

            result = await web_search("test query")

        assert result["status"] == "error"
        assert "No results found" in result["result"]

    @pytest.mark.asyncio
    async def test_web_search_none_response(self) -> None:
        """Test handling of None response from API."""
        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=None)
            mock_get_client.return_value = mock_client

            result = await web_search("test query")

        assert result["status"] == "error"
        assert "No results found" in result["result"]

    @pytest.mark.asyncio
    async def test_web_search_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_get_client.return_value = mock_client

            result = await web_search("test query")

        assert result["status"] == "error"
        assert "Search failed" in result["result"]
        assert "API Error" in result["result"]

    @pytest.mark.asyncio
    async def test_web_search_uses_configured_model(self) -> None:
        """Test that search uses the configured model."""
        mock_response = MagicMock()
        mock_response.text = "Result"
        mock_response.candidates = []

        with patch("app.tools.web_search_tool._get_client") as mock_get_client:
            with patch("app.tools.web_search_tool.os.getenv") as mock_getenv:
                mock_getenv.side_effect = lambda key, default="": {
                    "SEARCH_MODEL_NAME": "custom-flash-model",
                    "VERTEX_PROJECT_ID": "test",
                    "VERTEX_LOCATION": "us-central1",
                }.get(key, default)

                mock_client = MagicMock()
                mock_client.aio.models.generate_content = AsyncMock(
                    return_value=mock_response
                )
                mock_get_client.return_value = mock_client

                await web_search("test")

                # Verify the model was passed correctly
                call_args = mock_client.aio.models.generate_content.call_args
                assert call_args[1]["model"] == "custom-flash-model"

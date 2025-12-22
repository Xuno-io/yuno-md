"""
Unit tests for NeibotService with Google ADK integration.

Tests cover the ADK-based agent execution, tool creation, session management,
memory capture, and response distillation.
"""

import logging
from app.services.NeibotService.neibot_service import (
    NeibotService,
    _current_user_context,
)
from typing import cast
from unittest.mock import AsyncMock, patch, MagicMock, Mock

import pytest

from app.entities.message import MessagePayload


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    return logging.getLogger("NeibotServiceTest")


@pytest.fixture
def neibot_service(logger: logging.Logger) -> NeibotService:
    """Create a NeibotService instance for testing."""
    with patch(
        "app.services.NeibotService.neibot_service.genai.Client"
    ) as MockGenAIClient:
        with patch(
            "app.services.NeibotService.neibot_service.InMemorySessionService"
        ) as MockSessionService:
            # Setup the mock genai client for distillation
            mock_genai_instance = MockGenAIClient.return_value
            mock_genai_instance.aio.models.generate_content = AsyncMock()

            # Setup the mock session service
            mock_session_service_instance = MockSessionService.return_value
            mock_session_service_instance.create_session = AsyncMock()
            mock_session_service_instance.append_event = AsyncMock()

            service = NeibotService(
                system_prompt="You are a helpful assistant.",
                model_name="gemini-pro",
                location="us-central1",
                project_id="test-project",
                temperature=0.7,
                max_tokens=1000,
                logger=logger,
                cache_threshold=2048,
            )
            return service


class TestNeibotServiceInitialization:
    """Test cases for NeibotService initialization."""

    def test_initialization_sets_correct_attributes(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that all attributes are set correctly during initialization."""
        assert neibot_service.system_prompt == "You are a helpful assistant."
        assert neibot_service.model_name == "gemini-pro"
        assert neibot_service.location == "us-central1"
        assert neibot_service.project_id == "test-project"
        assert neibot_service.temperature == 0.7
        assert neibot_service.max_tokens == 1000

    def test_default_model_names(self, logger: logging.Logger) -> None:
        """Test that default model names are set correctly."""
        with patch("app.services.NeibotService.neibot_service.genai.Client"):
            with patch(
                "app.services.NeibotService.neibot_service.InMemorySessionService"
            ):
                service = NeibotService(
                    system_prompt="Test",
                    model_name="gemini-pro",
                    location="us-central1",
                    project_id="test-project",
                    temperature=0.7,
                    max_tokens=1000,
                    logger=logger,
                )

        assert service.extraction_model_name == "gemini-2.0-flash"
        assert service.distill_model_name == "gemini-2.5-pro"

    def test_custom_model_names(self, logger: logging.Logger) -> None:
        """Test that custom model names are used when provided."""
        with patch("app.services.NeibotService.neibot_service.genai.Client"):
            with patch(
                "app.services.NeibotService.neibot_service.InMemorySessionService"
            ):
                service = NeibotService(
                    system_prompt="Test",
                    model_name="gemini-pro",
                    location="us-central1",
                    project_id="test-project",
                    temperature=0.7,
                    max_tokens=1000,
                    logger=logger,
                    extraction_model_name="custom-extraction",
                    distill_model_name="custom-distill",
                )

        assert service.extraction_model_name == "custom-extraction"
        assert service.distill_model_name == "custom-distill"


class TestCreateAgent:
    """Test cases for agent creation."""

    def test_create_agent_includes_web_search_tool(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that web_search tool is included in the agent."""
        with patch("app.services.NeibotService.neibot_service.Agent") as MockAgent:
            neibot_service._create_agent("gemini-pro")

            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args[1]
            tools = call_kwargs["tools"]

            # Should have at least web_search tool
            assert len(tools) >= 1
            # Check that web_search is in tools (it's a function)
            tool_names = [getattr(t, "__name__", str(t)) for t in tools]
            assert "web_search" in tool_names

    def test_create_agent_includes_memory_tool_when_service_available(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that search_memory tool is included when memory service exists."""
        mock_memory = MagicMock()
        neibot_service.memory_service = mock_memory

        with patch("app.services.NeibotService.neibot_service.Agent") as MockAgent:
            neibot_service._create_agent("gemini-pro")

            call_kwargs = MockAgent.call_args[1]
            tools = call_kwargs["tools"]

            # Should have both web_search and search_memory
            assert len(tools) == 2

    def test_create_agent_includes_time_in_instruction(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that current UTC time is included in the agent instruction."""
        with patch("app.services.NeibotService.neibot_service.Agent") as MockAgent:
            neibot_service._create_agent("gemini-pro")

            call_kwargs = MockAgent.call_args[1]
            instruction = call_kwargs["instruction"]

            assert "Current time (UTC):" in instruction


class TestSearchMemoryTool:
    """Test cases for the search_memory tool."""

    @pytest.mark.asyncio
    async def test_search_memory_success(self, neibot_service: NeibotService) -> None:
        """Test successful memory search execution."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = [
            {"memory": "[TECH_STACK] Uses Python"},
            {"memory": "[TECH_STACK] Uses Redis"},
        ]

        neibot_service.memory_service = mock_memory

        token = _current_user_context.set("user_123")
        try:
            # Get the tool function
            search_memory = neibot_service._create_search_memory_tool()

            result = await search_memory(query="tech stack", category="TECH_STACK")

            assert result["status"] == "success"
            assert "Uses Python" in result["result"]
            assert "Uses Redis" in result["result"]
        finally:
            _current_user_context.reset(token)

    @pytest.mark.asyncio
    async def test_search_memory_no_results(
        self, neibot_service: NeibotService
    ) -> None:
        """Test memory search with no results."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = []

        neibot_service.memory_service = mock_memory

        token = _current_user_context.set("user_123")
        try:
            search_memory = neibot_service._create_search_memory_tool()
            result = await search_memory(query="unknown topic", category="TECH_STACK")

            assert result["status"] == "success"
            assert "No relevant memories found" in result["result"]
        finally:
            _current_user_context.reset(token)

    @pytest.mark.asyncio
    async def test_search_memory_no_memory_service(
        self, neibot_service: NeibotService
    ) -> None:
        """Test memory search when no memory service available."""
        neibot_service.memory_service = None

        token = _current_user_context.set("user_123")
        try:
            search_memory = neibot_service._create_search_memory_tool()
            result = await search_memory(query="test", category="TECH_STACK")

            assert result["status"] == "error"
            assert "not available" in result["result"]
        finally:
            _current_user_context.reset(token)

    @pytest.mark.asyncio
    async def test_search_memory_no_user_id(
        self, neibot_service: NeibotService
    ) -> None:
        """Test memory search when no user context available."""
        mock_memory = MagicMock()
        neibot_service.memory_service = mock_memory

        # Ensure context is empty
        token = _current_user_context.set(None)
        try:
            search_memory = neibot_service._create_search_memory_tool()
            result = await search_memory(query="test", category="TECH_STACK")

            assert result["status"] == "error"
            assert "No user context" in result["result"]
        finally:
            _current_user_context.reset(token)

    @pytest.mark.asyncio
    async def test_search_memory_handles_exception(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test memory search handles exceptions."""
        mock_memory = MagicMock()
        mock_memory.search.side_effect = Exception("Search failed")

        neibot_service.memory_service = mock_memory

        token = _current_user_context.set("user_123")
        try:
            with patch.object(logger, "error") as mock_error:
                search_memory = neibot_service._create_search_memory_tool()
                result = await search_memory(query="test", category="TECH_STACK")

            assert result["status"] == "error"
            assert "failed" in result["result"].lower()
            mock_error.assert_called()
        finally:
            _current_user_context.reset(token)


class TestGetResponse:
    """Test cases for get_response with ADK."""

    @pytest.mark.asyncio
    async def test_get_response_empty_history(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test get_response with empty history returns empty string."""
        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response([])

        assert result == ""
        mock_warning.assert_called_with("Empty history provided")

    @pytest.mark.asyncio
    async def test_get_response_success(self, neibot_service: NeibotService) -> None:
        """Test successful response from ADK agent."""
        # Create mock session
        mock_session = MagicMock()
        mock_session.id = "test-session-id"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        # Create mock event with final response
        mock_event = MagicMock()
        mock_event.is_final_response.return_value = True
        mock_content = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Hello! How can I help you?"
        mock_content.parts = [mock_part]
        mock_event.content = mock_content

        # Setup the runner mock
        with patch("app.services.NeibotService.neibot_service.Agent"):
            with patch(
                "app.services.NeibotService.neibot_service.Runner"
            ) as MockRunner:
                mock_runner = MockRunner.return_value

                # Create an async generator for run_async
                async def mock_run_async(*args, **kwargs):
                    yield mock_event

                mock_runner.run_async = mock_run_async

                history: list[MessagePayload] = [
                    {"role": "user", "content": "Hello", "attachments": []}
                ]

                result = await neibot_service.get_response(history)

        assert result == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_get_response_with_custom_model(
        self, neibot_service: NeibotService
    ) -> None:
        """Test get_response uses custom model when provided."""
        mock_session = MagicMock()
        mock_session.id = "test-session-id"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        # Create mock event with empty response
        mock_event = MagicMock()
        mock_event.is_final_response.return_value = True
        mock_event.content = None

        with patch("app.services.NeibotService.neibot_service.Agent") as MockAgent:
            with patch(
                "app.services.NeibotService.neibot_service.Runner"
            ) as MockRunner:
                mock_runner = MockRunner.return_value

                async def mock_run_async(*args, **kwargs):
                    yield mock_event

                mock_runner.run_async = mock_run_async

                history: list[MessagePayload] = [
                    {"role": "user", "content": "Hello", "attachments": []}
                ]

                await neibot_service.get_response(history, model_name="custom-model")

        # Verify Agent was created with custom model
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args[1]
        assert call_kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_get_response_sets_user_id_context(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that user_id is set for tool access during get_response."""
        mock_session = MagicMock()
        mock_session.id = "test-session-id"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        mock_event = MagicMock()
        mock_event.is_final_response.return_value = True
        mock_event.content = None

        with patch("app.services.NeibotService.neibot_service.Agent"):
            with patch(
                "app.services.NeibotService.neibot_service.Runner"
            ) as MockRunner:
                mock_runner = MockRunner.return_value

                async def mock_run_async(*args, **kwargs):
                    # Check that user_id was set during execution
                    assert _current_user_context.get() == "user_123"
                    yield mock_event

                mock_runner.run_async = mock_run_async

                history: list[MessagePayload] = [
                    {"role": "user", "content": "Hello", "attachments": []}
                ]

                await neibot_service.get_response(history, user_id="user_123")

        # User ID should be cleared after execution (back to None)
        assert _current_user_context.get() is None

    @pytest.mark.asyncio
    async def test_get_response_handles_exception(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling of exceptions during agent execution."""
        cast(
            Mock, neibot_service.session_service.create_session
        ).side_effect = Exception("Session error")

        with patch.object(logger, "error") as mock_error:
            history: list[MessagePayload] = [
                {"role": "user", "content": "Hello", "attachments": []}
            ]

            result = await neibot_service.get_response(history)

        assert "couldn't process your request" in result
        mock_error.assert_called()

    @pytest.mark.asyncio
    async def test_get_response_handles_empty_user_message_content(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that empty user message content falls back to placeholder."""
        mock_session = MagicMock()
        mock_session.id = "test-session-id"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        # Mock event (response)
        mock_event = MagicMock()
        mock_event.is_final_response.return_value = True
        mock_content = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "I see nothing."
        mock_content.parts = [mock_part]
        mock_event.content = mock_content

        with patch("app.services.NeibotService.neibot_service.Agent"):
            with patch(
                "app.services.NeibotService.neibot_service.Runner"
            ) as MockRunner:
                mock_runner = MockRunner.return_value

                async def mock_run_async(*args, **kwargs):
                    # Capture the new_message passed to runner
                    new_message = kwargs.get("new_message")
                    assert new_message is not None
                    # Verify the fix: empty content replaced by placeholder
                    assert len(new_message.parts) == 1
                    # Access text attribute directly as it's a real Part object
                    assert "[System note:" in new_message.parts[0].text
                    yield mock_event

                mock_runner.run_async = mock_run_async

                # History with empty content ("") and empty attachments
                history: list[MessagePayload] = [
                    {"role": "user", "content": "", "attachments": []}
                ]

                # We need to ensure we catch the warning
                with patch.object(logger, "warning") as mock_warning:
                    await neibot_service.get_response(history)

                    found = False
                    # Check arguments of all warning calls
                    for call in mock_warning.call_args_list:
                        args, _ = call
                        if args and "new_message parts empty" in args[0]:
                            found = True
                            break
                    assert (
                        found
                    ), "Expected warning about empty message parts not found."


class TestCreateSessionWithHistory:
    """Test cases for session creation with history."""

    @pytest.mark.asyncio
    async def test_creates_session_with_user_id(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that session is created with correct user_id."""
        mock_session = MagicMock()
        mock_session.id = "test-session"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        await neibot_service._create_session_with_history(
            user_id="test_user",
            history=[],
        )

        cast(Mock, neibot_service.session_service.create_session).assert_called_once()
        call_kwargs = cast(
            Mock, neibot_service.session_service.create_session
        ).call_args[1]
        assert call_kwargs["user_id"] == "test_user"
        assert call_kwargs["app_name"] == "yunoai"

    @pytest.mark.asyncio
    async def test_appends_history_events(self, neibot_service: NeibotService) -> None:
        """Test that history messages are appended as events."""
        mock_session = MagicMock()
        mock_session.id = "test-session"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        history: list[MessagePayload] = [
            {"role": "user", "content": "First message", "attachments": []},
            {"role": "assistant", "content": "First response", "attachments": []},
            {"role": "user", "content": "Second message", "attachments": []},
        ]

        await neibot_service._create_session_with_history(
            user_id="test_user",
            history=history,
        )

        # Should append 2 events (history minus the last message)
        assert cast(Mock, neibot_service.session_service.append_event).call_count == 2

    @pytest.mark.asyncio
    async def test_skips_system_messages(self, neibot_service: NeibotService) -> None:
        """Test that system messages are skipped when building history."""
        mock_session = MagicMock()
        mock_session.id = "test-session"
        cast(
            Mock, neibot_service.session_service.create_session
        ).return_value = mock_session

        history: list[MessagePayload] = [
            {"role": "system", "content": "System prompt", "attachments": []},
            {"role": "user", "content": "User message", "attachments": []},
            {"role": "user", "content": "Latest message", "attachments": []},
        ]

        await neibot_service._create_session_with_history(
            user_id="test_user",
            history=history,
        )

        # Should only append 1 event (skipping system and last message)
        assert cast(Mock, neibot_service.session_service.append_event).call_count == 1


class TestCaptureFactsFromHistory:
    """Test cases for capture_facts_from_history method."""

    @pytest.mark.asyncio
    async def test_capture_no_memory_service(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test capture returns 0 when no memory service."""
        neibot_service.memory_service = None

        history: list[MessagePayload] = [
            {"role": "user", "content": "I use Python", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.capture_facts_from_history(
                history, user_id="user_123"
            )

        assert result == 0
        mock_warning.assert_called()

    @pytest.mark.asyncio
    async def test_capture_with_memory_service(
        self, neibot_service: NeibotService
    ) -> None:
        """Test capture delegates to memory service."""
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [{"id": "mem_1"}, {"id": "mem_2"}]}
        neibot_service.memory_service = mock_memory

        history: list[MessagePayload] = [
            {"role": "user", "content": "I use Python and Redis", "attachments": []}
        ]

        result = await neibot_service.capture_facts_from_history(
            history, user_id="user_123"
        )

        assert result == 2
        mock_memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_empty_history(self, neibot_service: NeibotService) -> None:
        """Test capture with empty history returns 0."""
        mock_memory = MagicMock()
        neibot_service.memory_service = mock_memory

        history: list[MessagePayload] = []

        result = await neibot_service.capture_facts_from_history(
            history, user_id="user_123"
        )

        assert result == 0

    @pytest.mark.asyncio
    async def test_capture_handles_exception(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test capture handles exceptions gracefully."""
        mock_memory = MagicMock()
        mock_memory.add.side_effect = Exception("Memory error")
        neibot_service.memory_service = mock_memory

        history: list[MessagePayload] = [
            {"role": "user", "content": "Some content", "attachments": []}
        ]

        with patch.object(logger, "error") as mock_error:
            result = await neibot_service.capture_facts_from_history(
                history, user_id="user_123"
            )

        assert result == 0
        mock_error.assert_called()


class TestDistillResponse:
    """Test cases for the distill_response method."""

    @pytest.mark.asyncio
    async def test_distill_response_success(
        self, neibot_service: NeibotService
    ) -> None:
        """Test successful response distillation."""
        original_response = "A" * 5000  # Long response
        distilled = "1. EL ESTRATO: Core idea\n2. EL AGRIETAMIENTO: Exception\n3. LA LÍNEA DE FUGA: Mutation\n4. EL VECTOR: Action"

        mock_response = MagicMock()
        mock_response.text = distilled
        cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).return_value = mock_response

        result = await neibot_service.distill_response(original_response)

        assert result == distilled
        cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).assert_called_once()

    @pytest.mark.asyncio
    async def test_distill_response_with_context(
        self, neibot_service: NeibotService
    ) -> None:
        """Test distillation includes conversation context when provided."""
        original_response = "A" * 5000
        distilled = "Distilled with context"
        context: list[MessagePayload] = [
            {"role": "user", "content": "What is Python?", "attachments": []},
            {"role": "assistant", "content": "Python is a language", "attachments": []},
            {"role": "user", "content": "Tell me more", "attachments": []},
        ]

        mock_response = MagicMock()
        mock_response.text = distilled
        cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).return_value = mock_response

        result = await neibot_service.distill_response(
            original_response, context=context
        )

        assert result == distilled
        call_args = cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).call_args
        contents = call_args[1]["contents"]
        prompt_text = contents[0].parts[0].text
        assert "CONTEXTO DE LA CONVERSACIÓN" in prompt_text
        assert "What is Python?" in prompt_text

    @pytest.mark.asyncio
    async def test_distill_response_empty_result(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling when distillation returns empty response."""
        original_response = "A" * 5000

        mock_response = MagicMock()
        mock_response.text = ""
        cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).return_value = mock_response

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.distill_response(original_response)

        assert "Error" in result or "No pude" in result
        mock_warning.assert_called()

    @pytest.mark.asyncio
    async def test_distill_response_handles_exception(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling of exceptions during distillation."""
        original_response = "A" * 5000

        cast(
            Mock, neibot_service.client.aio.models.generate_content
        ).side_effect = Exception("API Error")

        with patch.object(logger, "error") as mock_error:
            result = await neibot_service.distill_response(original_response)

        assert "Error" in result
        mock_error.assert_called()

    def test_load_distill_prompt_file_not_found(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test fallback to default prompt when file not found."""
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError()):
            with patch.object(logger, "warning") as mock_warning:
                prompt = neibot_service._load_distill_prompt()

        assert "EL ESTRATO" in prompt or "CUATRO movimientos" in prompt
        mock_warning.assert_called()

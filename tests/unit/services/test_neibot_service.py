import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service import NeibotService


@pytest.fixture
def logger() -> logging.Logger:
    """Create a test logger."""
    return logging.getLogger("NeibotServiceTest")


@pytest.fixture
def neibot_service(logger: logging.Logger) -> NeibotService:
    """Create a NeibotService instance for testing."""
    with patch("app.services.NeibotService.neibot_service.AsyncOpenAI"):
        service = NeibotService(
            system_prompt="You are a helpful assistant.",
            model_name="gpt-4",
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            temperature=0.7,
            max_tokens=1000,
            logger=logger,
            cache_threshold=2048,
        )
        # Mock the client to avoid actual API calls
        service.client = AsyncMock()
        return service


class TestBuildMessagesAttachmentHandling:
    """Test cases for safe attachment handling in _build_messages."""

    def test_build_messages_skips_attachment_without_base64(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that attachments without base64 are skipped with a warning."""
        # Intentionally malformed attachment to test error handling
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "file_name": "image.png",
                            # Missing base64 key - intentionally malformed
                        }
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        # Should have system message and user message with only text (no image)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 1  # Only text, no image
        assert messages[1]["content"][0]["type"] == "text"
        mock_warning.assert_called_once()
        assert "invalid or missing base64 data" in str(mock_warning.call_args)

    def test_build_messages_skips_attachment_with_empty_base64(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that attachments with empty base64 are skipped."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": "",
                            "file_name": "image.png",
                        }
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 1  # Only text, no image
        mock_warning.assert_called_once()

    def test_build_messages_skips_attachment_with_whitespace_base64(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that attachments with whitespace-only base64 are skipped."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": "   \n\t  ",
                            "file_name": "image.png",
                        }
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 1  # Only text, no image
        mock_warning.assert_called_once()

    def test_build_messages_skips_attachment_with_none_base64(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that attachments with None base64 are skipped."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": None,  # type: ignore[dict-item]
                            "file_name": "image.png",
                        }
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 1  # Only text, no image
        mock_warning.assert_called_once()

    def test_build_messages_skips_attachment_with_non_string_base64(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that attachments with non-string base64 are skipped."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": 12345,  # type: ignore[dict-item]  # Not a string
                            "file_name": "image.png",
                        }
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 1  # Only text, no image
        mock_warning.assert_called_once()

    def test_build_messages_includes_valid_attachment(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that valid attachments are included in the message."""
        history: list[MessagePayload] = [
            {
                "role": "user",
                "content": "Look at this image",
                "attachments": [
                    {
                        "mime_type": "image/png",
                        "size_bytes": 100,
                        "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        "file_name": "image.png",
                    }
                ],
            }
        ]

        messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 2  # Text + image
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in messages[1]["content"][1]["image_url"]["url"]

    def test_build_messages_uses_default_mime_type(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that default mime_type is used when not provided."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "size_bytes": 100,
                            "base64": "YWJjZGVmZw==",
                            "file_name": "image.jpg",
                        }
                    ],
                },
            )
        ]

        messages = neibot_service._build_messages(history)

        assert len(messages) == 2
        image_url = messages[1]["content"][1]["image_url"]["url"]
        assert "data:image/jpeg;base64," in image_url

    def test_build_messages_processes_multiple_attachments_with_invalid_ones(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that valid attachments are processed even when some are invalid."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at these images",
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": "",  # Invalid: empty
                            "file_name": "image1.png",
                        },
                        {
                            "mime_type": "image/jpeg",
                            "size_bytes": 100,
                            "base64": "YWJjZGVmZw==",  # Valid
                            "file_name": "image2.jpg",
                        },
                        {
                            "mime_type": "image/gif",
                            "size_bytes": 100,
                            # Missing base64 - intentionally malformed
                            "file_name": "image3.gif",
                        },
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        # Should have text + 1 valid image (2 invalid ones skipped)
        assert len(messages) == 2
        assert isinstance(messages[1]["content"], list)
        assert len(messages[1]["content"]) == 2  # Text + 1 image
        assert messages[1]["content"][1]["type"] == "image_url"
        assert (
            "data:image/jpeg;base64," in messages[1]["content"][1]["image_url"]["url"]
        )
        # Should have logged 2 warnings (one for each invalid attachment)
        assert mock_warning.call_count == 2

    def test_build_messages_skips_message_with_empty_content_parts(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that message is skipped when no text AND all attachments are invalid."""
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "",  # Empty text
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "size_bytes": 100,
                            "base64": "",  # Invalid: empty base64
                            "file_name": "image1.png",
                        },
                        {
                            "mime_type": "image/jpeg",
                            "size_bytes": 100,
                            # Missing base64 - intentionally malformed
                            "file_name": "image2.jpg",
                        },
                    ],
                },
            )
        ]

        with patch.object(logger, "warning") as mock_warning:
            messages = neibot_service._build_messages(history)

        # Should only have system message (user message skipped)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        # Should have logged warnings for invalid attachments + one for skipped message
        assert mock_warning.call_count == 3
        # Verify the skip message was logged
        warning_calls = [str(call) for call in mock_warning.call_args_list]
        assert any("empty content" in call for call in warning_calls)

    def test_build_messages_skips_message_preserves_other_messages(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that skipping one message preserves valid messages in history."""
        history: list[MessagePayload] = [
            {"role": "user", "content": "First message", "attachments": []},
            cast(
                MessagePayload,
                {
                    "role": "assistant",
                    "content": "",  # Empty text
                    "attachments": [
                        {
                            "mime_type": "image/png",
                            "base64": "",  # Invalid
                            "file_name": "broken.png",
                        }
                    ],
                },
            ),
            {"role": "user", "content": "Third message", "attachments": []},
        ]

        with patch.object(logger, "warning"):
            messages = neibot_service._build_messages(history)

        # Should have: system + first user + third user (assistant skipped)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "First message"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Third message"


class TestGetResponseMalformedHandling:
    """Test cases for safe handling of malformed API responses in get_response."""

    @pytest.mark.asyncio
    async def test_get_response_handles_empty_response(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that empty response is handled safely."""
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=None),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with("Received empty response from API")

    @pytest.mark.asyncio
    async def test_get_response_handles_response_without_choices(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that response without choices attribute is handled safely."""

        # Create an object without choices attribute (using object() which has no attributes)
        class ResponseWithoutChoices:
            pass

        mock_response = ResponseWithoutChoices()
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with(
            "Response has no choices or choices list is empty"
        )

    @pytest.mark.asyncio
    async def test_get_response_handles_empty_choices_list(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that response with empty choices list is handled safely."""
        mock_response = SimpleNamespace(choices=[])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with(
            "Response has no choices or choices list is empty"
        )

    @pytest.mark.asyncio
    async def test_get_response_handles_choice_without_message(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that choice without message attribute is handled safely."""

        # Create a choice object without message attribute
        class ChoiceWithoutMessage:
            pass

        mock_choice = ChoiceWithoutMessage()
        mock_response = SimpleNamespace(choices=[mock_choice])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with(
            "First choice is missing or has no message attribute"
        )

    @pytest.mark.asyncio
    async def test_get_response_handles_message_without_content(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that message without content attribute is handled safely."""

        # Create a message object without content attribute
        class MessageWithoutContent:
            pass

        mock_message = MessageWithoutContent()
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(choices=[mock_choice])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with(
            "Message has no content attribute or content is None"
        )

    @pytest.mark.asyncio
    async def test_get_response_handles_message_with_none_content(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that message with None content is handled safely."""
        mock_message = SimpleNamespace(content=None)
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(choices=[mock_choice])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_once_with(
            "Message has no content attribute or content is None"
        )

    @pytest.mark.asyncio
    async def test_get_response_handles_empty_content_string(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that empty content string returns empty string."""
        mock_message = SimpleNamespace(content="")
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(choices=[mock_choice])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        result = await neibot_service.get_response(history)

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_response_returns_valid_content(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that valid response content is returned correctly."""
        mock_message = SimpleNamespace(content="Hello, how can I help you?")
        mock_choice = SimpleNamespace(message=mock_message)
        mock_response = SimpleNamespace(choices=[mock_choice])
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(return_value=mock_response),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        result = await neibot_service.get_response(history)

        assert result == "Hello, how can I help you?"

    @pytest.mark.asyncio
    async def test_get_response_handles_exception_gracefully(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test that exceptions are caught and handled gracefully."""
        setattr(
            neibot_service.client.chat.completions,
            "create",
            AsyncMock(side_effect=Exception("API Error")),
        )
        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "error") as mock_error:
            result = await neibot_service.get_response(history)

        assert result == "I'm sorry, I couldn't process your request at the moment."
        mock_error.assert_called_once()
        assert "Error getting response from OpenAI" in str(mock_error.call_args)

import logging
import base64
from typing import cast
from unittest.mock import AsyncMock, patch, MagicMock

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
    with patch("app.services.NeibotService.neibot_service.genai.Client") as MockClient:
        # Setup the mock client structure
        mock_client_instance = MockClient.return_value
        mock_client_instance.aio.models.generate_content = AsyncMock()

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


class TestBuildContentsAttachmentHandling:
    """Test cases for safe attachment handling in _build_contents."""

    def test_build_contents_skips_attachment_without_base64(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that attachments without base64 are skipped with a warning."""
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
                            # Missing base64 key
                        }
                    ],
                },
            )
        ]

        with patch.object(neibot_service.logger, "warning") as mock_warning:
            contents = neibot_service._build_contents(history)

        # Should have one user content (system is handled in config now)
        # The user content should only have the text part
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == "Look at this image"

        # Verify warning was called for missing base64
        mock_warning.assert_called_once()
        assert "missing base64 data" in str(mock_warning.call_args).lower()

    def test_build_contents_skips_attachment_with_empty_base64(
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

        contents = neibot_service._build_contents(history)

        assert len(contents) == 1
        assert len(contents[0].parts) == 1  # Only text
        assert contents[0].parts[0].text == "Look at this image"

    def test_build_contents_includes_valid_attachment(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that valid attachments are included in the message."""
        valid_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        history: list[MessagePayload] = [
            {
                "role": "user",
                "content": "Look at this image",
                "attachments": [
                    {
                        "mime_type": "image/png",
                        "size_bytes": 100,
                        "base64": valid_b64,
                        "file_name": "image.png",
                    }
                ],
            }
        ]

        contents = neibot_service._build_contents(history)

        assert len(contents) == 1
        assert len(contents[0].parts) == 2  # Text + image

        # Check text part
        assert contents[0].parts[0].text == "Look at this image"

        # Check image part
        # Since we can't easily inspect the Part object internal bytes without knowing the exact structure (it might be in inline_data),
        # we assume the library constructs it correctly if we passed it.
        # But we can check if it's there.
        image_part = contents[0].parts[1]
        # Verify it's not text
        assert image_part.text is None
        # Verify it has inline_data (Blob)
        assert image_part.inline_data is not None
        assert image_part.inline_data.mime_type == "image/png"
        assert image_part.inline_data.data == base64.b64decode(valid_b64)

    def test_build_contents_uses_default_mime_type(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that default mime_type is used when not provided."""
        valid_b64 = "YWJjZGVmZw=="
        history: list[MessagePayload] = [
            cast(
                MessagePayload,
                {
                    "role": "user",
                    "content": "Look at this image",
                    "attachments": [
                        {
                            "size_bytes": 100,
                            "base64": valid_b64,
                            "file_name": "image.jpg",
                            # Missing mime_type
                        }
                    ],
                },
            )
        ]

        contents = neibot_service._build_contents(history)

        assert len(contents) == 1
        assert len(contents[0].parts) == 2
        image_part = contents[0].parts[1]
        assert image_part.inline_data.mime_type == "image/jpeg"  # Default

    def test_build_contents_handles_system_and_assistant_roles(
        self, neibot_service: NeibotService
    ) -> None:
        """Test that roles are mapped correctly."""
        history: list[MessagePayload] = [
            {"role": "system", "content": "System prompt", "attachments": []},
            {"role": "user", "content": "User msg", "attachments": []},
            {"role": "assistant", "content": "Assistant msg", "attachments": []},
        ]

        contents = neibot_service._build_contents(history)

        # System message should be skipped (handled in config)
        # User -> user
        # Assistant -> model
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "User msg"

        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Assistant msg"


class TestGetResponseHandling:
    """Test cases for get_response."""

    @pytest.mark.asyncio
    async def test_get_response_success(self, neibot_service: NeibotService) -> None:
        """Test successful response."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = "Response text"
        neibot_service.client.aio.models.generate_content.return_value = mock_response

        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        result = await neibot_service.get_response(history)
        assert result == "Response text"

    @pytest.mark.asyncio
    async def test_get_response_empty_response(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling of empty response object."""
        neibot_service.client.aio.models.generate_content.return_value = None

        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert result == ""
        mock_warning.assert_called_with("Received empty response from Google Gen AI")

    @pytest.mark.asyncio
    async def test_get_response_safety_blocked(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling of safety blocked response."""
        mock_response = MagicMock()
        mock_response.text = ""  # No text because blocked

        candidate = MagicMock()
        candidate.finish_reason = "SAFETY"
        candidate.safety_ratings = ["some ratings"]

        mock_response.candidates = [candidate]

        neibot_service.client.aio.models.generate_content.return_value = mock_response

        history: list[MessagePayload] = [
            {"role": "user", "content": "Unsafe prompt", "attachments": []}
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = await neibot_service.get_response(history)

        assert "couldn't process your request due to safety" in result
        mock_warning.assert_called()

    @pytest.mark.asyncio
    async def test_get_response_exception(
        self, neibot_service: NeibotService, logger: logging.Logger
    ) -> None:
        """Test handling of exceptions during generation."""
        neibot_service.client.aio.models.generate_content.side_effect = Exception(
            "API Error"
        )

        history: list[MessagePayload] = [
            {"role": "user", "content": "Hello", "attachments": []}
        ]

        with patch.object(logger, "error") as mock_error:
            result = await neibot_service.get_response(history)

        assert "couldn't process your request at the moment" in result
        mock_error.assert_called()

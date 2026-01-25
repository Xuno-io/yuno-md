import asyncio
import base64
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from telethon.errors import MessageTooLongError

from app.entities.message import ImageAttachment, MessagePayload

from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service import (
    ImageDownloadError,
    ImageTooLargeError,
    TelegramService,
    UnsupportedImageError,
)
from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)
from app.services.UserService.user_service_interface import UserServiceInterface


class DummyNeibot(NeibotServiceInterface):
    memory_service: Any = None

    async def get_response(
        self,
        history: list[MessagePayload],
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> str:
        raise NotImplementedError

    async def capture_facts_from_history(
        self,
        history: list[MessagePayload],
        user_id: str,
    ) -> int:
        raise NotImplementedError

    async def distill_response(
        self,
        original_response: str,
        context: list[MessagePayload] | None = None,
    ) -> str:
        raise NotImplementedError


class StubMessage:
    def __init__(
        self,
        data: bytes | None = None,
        *,
        mime_type: str | None = None,
        file_name: str | None = None,
        has_photo: bool = False,
        download_error: Exception | None = None,
    ) -> None:
        self._data = data or b""
        self._download_error = download_error
        self.media = None
        self.photo = None

        if has_photo:
            self.photo = object()
            self.media = SimpleNamespace(photo=self.photo, document=None)
        elif mime_type is not None:
            attributes = []
            if file_name is not None:
                attributes.append(SimpleNamespace(file_name=file_name))
            document = SimpleNamespace(mime_type=mime_type, attributes=attributes)
            self.media = SimpleNamespace(document=document, photo=None)

    async def download_media(self, file):
        if self._download_error:
            raise self._download_error
        file.write(self._data)
        return file


@pytest.fixture
def telegram_service() -> TelegramService:
    # Provide simple stubs for chat_repository and admin_ids required by the
    # updated TelegramService constructor.
    chat_repo = cast(
        ChatRepositoryInterface,
        SimpleNamespace(
            get_model=lambda chat_id: None,
            set_model=lambda chat_id, model: None,
        ),
    )
    user_service = cast(
        UserServiceInterface,
        SimpleNamespace(
            get_user_max_history_turns=lambda user_id: 100,
            set_user_pro_status=lambda user_id, is_pro: None,
            is_user_pro=lambda user_id: False,
            get_user_model=lambda user_id: "gemini-3-flash-preview",
        ),
    )
    return TelegramService(
        command_prefix="/cmd",
        neibot=DummyNeibot(),
        telegram_client=SimpleNamespace(),
        logger=logging.getLogger("TelegramServiceTest"),
        chat_repository=chat_repo,
        user_service=user_service,
        admin_ids=[123],
    )


def test_collect_image_attachments_returns_data(
    telegram_service: TelegramService,
) -> None:
    payload = b"binary"
    message = StubMessage(data=payload, mime_type="image/jpeg", file_name="photo.jpg")

    attachments = asyncio.run(
        telegram_service._collect_image_attachments(message, strict=True)
    )

    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment["mime_type"] == "image/jpeg"
    assert attachment["file_name"] == "photo.jpg"
    assert base64.b64decode(attachment["base64"]) == payload


def test_collect_image_attachments_uses_photo_fallback(
    telegram_service: TelegramService,
) -> None:
    payload = b"photo-bytes"
    message = StubMessage(data=payload, has_photo=True)

    attachments = asyncio.run(
        telegram_service._collect_image_attachments(message, strict=True)
    )

    assert len(attachments) == 1
    assert attachments[0]["mime_type"] == "image/jpeg"
    assert base64.b64decode(attachments[0]["base64"]) == payload


def test_collect_image_attachments_raises_on_large_file(
    telegram_service: TelegramService,
) -> None:
    payload = b"a" * (TelegramService._MAX_IMAGE_BYTES + 1)
    message = StubMessage(data=payload, mime_type="image/png")

    with pytest.raises(ImageTooLargeError):
        asyncio.run(telegram_service._collect_image_attachments(message, strict=True))


def test_collect_image_attachments_rejects_non_image_strict(
    telegram_service: TelegramService,
) -> None:
    message = StubMessage(data=b"data", mime_type="application/pdf")

    with pytest.raises(UnsupportedImageError):
        asyncio.run(telegram_service._collect_image_attachments(message, strict=True))


def test_collect_image_attachments_skips_non_image_when_lenient(
    telegram_service: TelegramService,
) -> None:
    message = StubMessage(data=b"data", mime_type="application/pdf")

    attachments = asyncio.run(
        telegram_service._collect_image_attachments(message, strict=False)
    )

    assert attachments == []


def test_collect_image_attachments_returns_empty_without_media(
    telegram_service: TelegramService,
) -> None:
    message = StubMessage()

    attachments = asyncio.run(
        telegram_service._collect_image_attachments(message, strict=True)
    )

    assert attachments == []


def test_collect_image_attachments_wraps_download_error(
    telegram_service: TelegramService,
) -> None:
    message = StubMessage(
        download_error=RuntimeError("boom"),
        mime_type="image/png",
        data=b"data",
    )

    with pytest.raises(ImageDownloadError):
        asyncio.run(telegram_service._collect_image_attachments(message, strict=True))


def test_extract_referenced_messages_merges_trailing_attachments(
    telegram_service: TelegramService,
) -> None:
    history: list[MessagePayload] = [
        {
            "role": "user",
            "content": "[Group: Foo][User: Alice]: hola",
            "attachments": [],
        },
        {
            "role": "user",
            "content": "[Group: Foo][User: Bob]: mira esto",
            "attachments": [
                {
                    "mime_type": "image/png",
                    "size_bytes": 4,
                    "base64": "YWJjZA==",
                    "file_name": "foto.png",
                }
            ],
        },
    ]

    context, attachments = telegram_service._extract_referenced_messages(history)

    assert context == ["[Group: Foo][User: Bob]: mira esto"]
    assert attachments == [
        {
            "mime_type": "image/png",
            "size_bytes": 4,
            "base64": "YWJjZA==",
            "file_name": "foto.png",
        }
    ]
    assert history == [
        {
            "role": "user",
            "content": "[Group: Foo][User: Alice]: hola",
            "attachments": [],
        }
    ]


def test_extract_referenced_messages_stops_without_trailing_attachments(
    telegram_service: TelegramService,
) -> None:
    history: list[MessagePayload] = [
        {
            "role": "user",
            "content": "[Group: Foo][User: Alice]: hola",
            "attachments": [],
        }
    ]

    context, attachments = telegram_service._extract_referenced_messages(history)

    assert context == []
    assert attachments == []
    assert history == [
        {
            "role": "user",
            "content": "[Group: Foo][User: Alice]: hola",
            "attachments": [],
        }
    ]


def test_merge_attachment_lists_deduplicates(
    telegram_service: TelegramService,
) -> None:
    first: list[ImageAttachment] = [
        {
            "mime_type": "image/png",
            "size_bytes": 4,
            "base64": "YWJjZA==",
            "file_name": "foto.png",
        }
    ]
    second: list[ImageAttachment] = [
        {
            "mime_type": "image/png",
            "size_bytes": 4,
            "base64": "YWJjZA==",
            "file_name": "foto.png",
        },
        {
            "mime_type": "image/jpeg",
            "size_bytes": 5,
            "base64": "ZWZnaA==",
            "file_name": "foto.jpg",
        },
    ]

    merged = telegram_service._merge_attachment_lists(first, second)

    assert merged == [
        {
            "mime_type": "image/png",
            "size_bytes": 4,
            "base64": "YWJjZA==",
            "file_name": "foto.png",
        },
        {
            "mime_type": "image/jpeg",
            "size_bytes": 5,
            "base64": "ZWZnaA==",
            "file_name": "foto.jpg",
        },
    ]


class MockEvent:
    """Mock Telegram event for testing."""

    def __init__(self) -> None:
        self.replied_message: str | None = None
        self.sender_id: int = 0
        self.chat_id: int = 0
        self.reply_to_msg_id: int | None = None
        self.raw_text: str = ""
        self.id: int = 999  # Added for rolling window logic

    async def reply(self, message: str) -> None:
        self.replied_message = message


def test_handle_help_command(telegram_service: TelegramService) -> None:
    event = MockEvent()
    asyncio.run(telegram_service._handle_help_command(event))

    assert event.replied_message is not None
    assert "¡Hola! Soy Yuno." in event.replied_message
    assert "hilos de conversación" in event.replied_message


class TestHandleSaveCommand:
    """Tests for the /save command handler."""

    def test_save_command_no_memory_service(
        self, telegram_service: TelegramService
    ) -> None:
        """Test /save when memory service not available."""
        # Neibot has no memory_service
        cast(Any, telegram_service.neibot).memory_service = None
        cast(Any, telegram_service.neibot).capture_facts_from_history = AsyncMock(
            return_value=0
        )

        event = MockEvent()
        event.sender_id = 123
        event.chat_id = 456
        event.reply_to_msg_id = None
        event.raw_text = "/save"

        # Mock typing action context manager
        telegram_service.bot = SimpleNamespace(
            action=lambda chat_id, action: AsyncContextManager(),
            get_messages=AsyncMock(return_value=[]),
        )

        asyncio.run(telegram_service._handle_save_command(event))

        assert event.replied_message is not None
        assert (
            "No encontré nuevos datos" in event.replied_message
            or "Analizando" in event.replied_message
        )

    def test_save_command_with_saved_facts(
        self, telegram_service: TelegramService
    ) -> None:
        """Test /save when facts are saved."""
        cast(Any, telegram_service.neibot).capture_facts_from_history = AsyncMock(
            return_value=3
        )

        event = MockEvent()
        event.sender_id = 123
        event.chat_id = 456
        event.reply_to_msg_id = None
        event.raw_text = "/save"

        telegram_service.bot = SimpleNamespace(
            action=lambda chat_id, action: AsyncContextManager(),
            get_messages=AsyncMock(return_value=[]),
        )

        asyncio.run(telegram_service._handle_save_command(event))

        # Should have at least started the process
        assert event.replied_message is not None


class TestHandleMemoryCommand:
    """Tests for the /memory command handler."""

    def test_memory_command_no_service(self, telegram_service: TelegramService) -> None:
        """Test /memory when no memory service available."""
        cast(Any, telegram_service.neibot).memory_service = None

        event = MockEvent()
        event.sender_id = 123
        event.raw_text = "/memory"

        asyncio.run(telegram_service._handle_memory_command(event))

        assert event.replied_message is not None
        assert "no está disponible" in event.replied_message

    def test_memory_command_empty_memories(
        self, telegram_service: TelegramService
    ) -> None:
        """Test /memory when user has no memories."""
        mock_memory_service = SimpleNamespace(
            get_all=lambda user_id: [],
            delete_all=lambda user_id: 0,
        )
        cast(Any, telegram_service.neibot).memory_service = mock_memory_service

        event = MockEvent()
        event.sender_id = 123
        event.raw_text = "/memory"

        asyncio.run(telegram_service._handle_memory_command(event))

        assert event.replied_message is not None
        assert "No tengo memorias guardadas" in event.replied_message

    def test_memory_command_with_memories(
        self, telegram_service: TelegramService
    ) -> None:
        """Test /memory when user has memories."""
        mock_memory_service = SimpleNamespace(
            get_all=lambda user_id: [
                {"memory": "[TECH_STACK] Uses Python"},
                {"memory": "[BUSINESS_LOGIC] Budget is 500"},
            ],
        )
        cast(Any, telegram_service.neibot).memory_service = mock_memory_service

        event = MockEvent()
        event.sender_id = 123
        event.raw_text = "/memory"

        asyncio.run(telegram_service._handle_memory_command(event))

        assert event.replied_message is not None
        assert "Tu memoria en Yuno" in event.replied_message
        assert "Stack Técnico" in event.replied_message

    def test_memory_clear_command(self, telegram_service: TelegramService) -> None:
        """Test /memory clear command."""
        mock_memory_service = SimpleNamespace(
            delete_all=lambda user_id: 5,
        )
        cast(Any, telegram_service.neibot).memory_service = mock_memory_service

        event = MockEvent()
        event.sender_id = 123
        event.raw_text = "/memory clear"

        asyncio.run(telegram_service._handle_memory_command(event))

        assert event.replied_message is not None
        assert "borrado" in event.replied_message
        assert "5" in event.replied_message


class AsyncContextManager:
    """Async context manager stub for bot.action."""

    async def __aenter__(self) -> "AsyncContextManager":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class MockEventWithDistillation:
    """Mock Telegram event that can simulate MessageTooLongError."""

    def __init__(
        self, fail_first_reply: bool = False, fail_second_reply: bool = False
    ) -> None:
        self.replied_messages: list[str] = []
        self.sender_id: int = 123
        self.chat_id: int = 456
        self.reply_to_msg_id: int | None = None
        self.raw_text: str = ""
        self.id: int = 888  # Added for rolling window logic
        self._fail_first_reply = fail_first_reply
        self._fail_second_reply = fail_second_reply
        self._reply_count = 0

    async def reply(self, message: str) -> None:
        self._reply_count += 1
        if self._fail_first_reply and self._reply_count == 1:
            raise MessageTooLongError(request=None)
        if self._fail_second_reply and self._reply_count == 2:
            raise MessageTooLongError(request=None)
        self.replied_messages.append(message)

    async def get_chat(self) -> SimpleNamespace:
        return SimpleNamespace(title="TestGroup", username=None)

    async def get_sender(self) -> SimpleNamespace:
        return SimpleNamespace(first_name="Test", last_name="User", username="testuser")


class TestDistillationProtocol:
    """Tests for the message distillation protocol when responses are too long."""

    def test_distillation_triggered_on_message_too_long(
        self, telegram_service: TelegramService
    ) -> None:
        """Test that distillation is triggered when MessageTooLongError occurs."""
        long_response = "A" * 5000  # Response that exceeds Telegram limit
        distilled_response = "Distilled version of the response"

        # Setup mocks
        cast(Any, telegram_service.neibot).get_response = AsyncMock(
            return_value=long_response
        )
        cast(Any, telegram_service.neibot).distill_response = AsyncMock(
            return_value=distilled_response
        )

        event = MockEventWithDistillation(fail_first_reply=True)
        event.raw_text = "/cmd test message"

        telegram_service.bot = SimpleNamespace(
            action=lambda chat_id, action: AsyncContextManager(),
            get_messages=AsyncMock(return_value=None),
            get_me=AsyncMock(return_value=SimpleNamespace(id=999)),
        )
        telegram_service.me = SimpleNamespace(id=999)

        asyncio.run(telegram_service._my_event_handler(event))

        # Verify distill_response was called with the original long response and context
        cast(Any, telegram_service.neibot).distill_response.assert_called_once()
        call_args = cast(Any, telegram_service.neibot).distill_response.call_args
        assert call_args[0][0] == long_response  # First positional arg is the response
        assert call_args[1]["context"] is not None  # Context was passed

        # Verify the distilled response was sent
        assert len(event.replied_messages) == 1
        assert event.replied_messages[0] == distilled_response

    def test_no_distillation_when_response_fits(
        self, telegram_service: TelegramService
    ) -> None:
        """Test that distillation is NOT triggered when response fits in limit."""
        normal_response = "A short response"

        cast(Any, telegram_service.neibot).get_response = AsyncMock(
            return_value=normal_response
        )
        cast(Any, telegram_service.neibot).distill_response = AsyncMock()

        event = MockEventWithDistillation(fail_first_reply=False)
        event.raw_text = "/cmd test message"

        telegram_service.bot = SimpleNamespace(
            action=lambda chat_id, action: AsyncContextManager(),
            get_messages=AsyncMock(return_value=None),
            get_me=AsyncMock(return_value=SimpleNamespace(id=999)),
        )
        telegram_service.me = SimpleNamespace(id=999)

        asyncio.run(telegram_service._my_event_handler(event))

        # Verify distill_response was NOT called
        cast(Any, telegram_service.neibot).distill_response.assert_not_called()

        # Verify the normal response was sent
        assert len(event.replied_messages) == 1
        assert event.replied_messages[0] == normal_response

    def test_distilled_response_too_long_truncates(
        self, telegram_service: TelegramService
    ) -> None:
        """Test that when distilled response is also too long, it gets truncated."""
        long_response = "A" * 5000  # Response that exceeds Telegram limit
        # Create a distilled response that also exceeds the limit
        long_distilled_response = "B" * 5000

        # Setup mocks
        cast(Any, telegram_service.neibot).get_response = AsyncMock(
            return_value=long_response
        )
        cast(Any, telegram_service.neibot).distill_response = AsyncMock(
            return_value=long_distilled_response
        )

        # Mock event that fails both first and second reply
        event = MockEventWithDistillation(fail_first_reply=True, fail_second_reply=True)
        event.raw_text = "/cmd test message"

        telegram_service.bot = SimpleNamespace(
            action=lambda chat_id, action: AsyncContextManager(),
            get_messages=AsyncMock(return_value=None),
            get_me=AsyncMock(return_value=SimpleNamespace(id=999)),
        )
        telegram_service.me = SimpleNamespace(id=999)

        asyncio.run(telegram_service._my_event_handler(event))

        # Verify distill_response was called
        cast(Any, telegram_service.neibot).distill_response.assert_called_once()

        # Verify the truncated response was sent (should have truncation notice)
        assert len(event.replied_messages) == 1
        final_message = event.replied_messages[0]
        assert "[... mensaje truncado por límite de caracteres]" in final_message
        # Verify it's within Telegram's limit (4096 chars)
        assert len(final_message) <= 4096

import asyncio
import base64
import logging
from types import SimpleNamespace

import pytest
from typing import cast

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
    def __init__(self):
        self.replied_message = None

    async def reply(self, message):
        self.replied_message = message


def test_handle_help_command(telegram_service: TelegramService) -> None:
    event = MockEvent()
    asyncio.run(telegram_service._handle_help_command(event))

    assert "¡Hola! Soy Yuno." in event.replied_message
    assert "hilos de conversación" in event.replied_message

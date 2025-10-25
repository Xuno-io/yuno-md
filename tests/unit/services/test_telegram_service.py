import asyncio
import base64
import logging
from types import SimpleNamespace

import pytest

from app.entities.message import ImageAttachment, MessagePayload

from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service import (
    ImageDownloadError,
    ImageTooLargeError,
    TelegramService,
    UnsupportedImageError,
)


class DummyNeibot(NeibotServiceInterface):
    def get_response(self, history):
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
    return TelegramService(
        command_prefix="/cmd",
        neibot=DummyNeibot(),
        telegram_client=SimpleNamespace(),
        logger=logging.getLogger("TelegramServiceTest"),
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

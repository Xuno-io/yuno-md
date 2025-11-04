from __future__ import annotations

import asyncio
import base64
import logging
from io import BytesIO
from typing import Any, Literal, cast

from telethon import TelegramClient, events

from app.entities.message import ImageAttachment, MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)


class ImageProcessingError(Exception):
    """Base error for attachment processing failures."""


class ImageTooLargeError(ImageProcessingError):
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes
        super().__init__(f"Attachment exceeds size limit: {size_bytes} bytes")


class UnsupportedImageError(ImageProcessingError):
    def __init__(self, mime_type: str | None) -> None:
        self.mime_type = mime_type
        super().__init__(f"Unsupported mime type: {mime_type}")


class ImageDownloadError(ImageProcessingError):
    """Raised when a Telegram attachment cannot be downloaded."""


class TelegramService(TelegramServiceInterface):
    _MAX_IMAGE_BYTES: int = 5 * 1024 * 1024

    def __init__(
        self,
        command_prefix: str,
        neibot: NeibotServiceInterface,
        telegram_client: TelegramClient,
        logger: logging.Logger,
    ) -> None:
        self.logger: logging.Logger = logger
        self.neibot: NeibotServiceInterface = neibot
        self.bot: TelegramClient = telegram_client
        self.me: Any | None = None
        self.command_prefix: str = command_prefix

    @classmethod
    async def create(
        cls,
        command_prefix: str,
        neibot: NeibotServiceInterface,
        telegram_client: TelegramClient,
        logger: logging.Logger,
    ) -> TelegramService:
        """Factory to perform async setup steps before returning the service."""
        service = cls(
            command_prefix=command_prefix,
            neibot=neibot,
            telegram_client=telegram_client,
            logger=logger,
        )
        service.bot.add_event_handler(service._my_event_handler, events.NewMessage)
        return service

    async def _my_event_handler(self, event) -> None:
        if not await self._should_respond(event):
            self.logger.debug("Ignoring message: %s", event.raw_text)
            return

        await self.__ensure_identity()

        message = getattr(event, "message", None) or event

        try:
            attachments: list[ImageAttachment] = await self._collect_image_attachments(
                message, strict=True
            )
        except ImageTooLargeError as error:
            self.logger.warning("Received oversized image (%s bytes)", error.size_bytes)
            await event.reply(
                "La imagen supera el límite de 5 MB. Intenta con un archivo más liviano."
            )
            return
        except UnsupportedImageError as error:
            self.logger.info(
                "Unsupported attachment with mime type: %s",
                error.mime_type or "unknown",
            )
            await event.reply(
                "Solo puedo procesar imágenes en formatos compatibles (image/*)."
            )
            return
        except ImageDownloadError as error:
            self.logger.error("Failed to download image: %s", error)
            await event.reply(
                "No pude descargar la imagen. Intenta enviarla nuevamente."
            )
            return

        raw_message: str = event.raw_text or ""
        if raw_message.startswith(self.command_prefix):
            user_message = raw_message[len(self.command_prefix) :].strip()
        else:
            user_message = raw_message.strip()

        metadata: str = await self.__build_metadata(event)

        history: list[MessagePayload] = await self.__build_reply_history(event)

        context_texts, history_attachments = self._extract_referenced_messages(history)
        attachments = self._merge_attachment_lists(history_attachments, attachments)

        user_segment: str = f"{metadata}: {user_message}" if user_message else metadata

        filtered_context: list[str] = [
            text for text in context_texts if text and text != user_segment
        ]

        payload_parts: list[str] = filtered_context + [user_segment]
        payload: str = "\n".join(part for part in payload_parts if part)

        history.append(
            {
                "role": "user",
                "content": payload,
                "attachments": attachments,
            }
        )

        self.logger.info(
            "Constructed payload for Neibot with %s messages in history", len(history)
        )

        async with self.bot.action(event.chat_id, "typing"):
            response: str = await asyncio.to_thread(self.neibot.get_response, history)

        await event.reply(response)

    async def __build_metadata(self, event) -> str:
        chat = await event.get_chat()
        sender = await event.get_sender()
        chat_name: str = (
            getattr(chat, "title", None) or getattr(chat, "username", None) or "Direct"
        )
        sender_name_parts = [
            getattr(sender, "first_name", ""),
            getattr(sender, "last_name", ""),
        ]
        sender_name_candidates = [
            getattr(sender, "username", None),
            " ".join(part for part in sender_name_parts if part).strip(),
        ]
        sender_name: str = next(
            (name for name in sender_name_candidates if name), str(event.sender_id)
        )
        return f"[Group: {chat_name}][User: {sender_name}]"

    async def __build_reply_history(self, event) -> list[MessagePayload]:
        history: list[MessagePayload] = []
        current_event = event

        while current_event.reply_to_msg_id:
            try:
                replied_msg = await current_event.get_reply_message()
                if not replied_msg:
                    break

                replied_sender = await replied_msg.get_sender()
                replied_sender_name_parts = [
                    getattr(replied_sender, "first_name", ""),
                    getattr(replied_sender, "last_name", ""),
                ]
                replied_sender_name_candidates = [
                    getattr(replied_sender, "username", None),
                    " ".join(
                        part for part in replied_sender_name_parts if part
                    ).strip(),
                ]
                replied_sender_name: str = next(
                    (name for name in replied_sender_name_candidates if name),
                    str(replied_msg.sender_id),
                )

                await self.__ensure_identity()
                role: Literal["assistant", "user"] = (
                    "assistant"
                    if self.me and replied_msg.sender_id == self.me.id
                    else "user"
                )

                chat = await replied_msg.get_chat()
                chat_name: str = (
                    getattr(chat, "title", None)
                    or getattr(chat, "username", None)
                    or "Direct"
                )

                if role == "assistant":
                    message_content = replied_msg.raw_text or ""
                else:
                    message_content = f"[Group: {chat_name}][User: {replied_sender_name}]: {replied_msg.raw_text or ''}"

                attachments = await self._collect_image_attachments(
                    replied_msg, strict=False
                )

                history.insert(
                    0,
                    {
                        "role": role,
                        "content": message_content,
                        "attachments": attachments,
                    },
                )

                current_event = replied_msg

                self.logger.debug(
                    "Added message to history: %s - %s...",
                    role,
                    (message_content or "")[:50],
                )

            except Exception as exc:
                self.logger.error("Error fetching reply message: %s", exc)
                break

        self.logger.info("Built reply history with %s previous messages", len(history))
        return history

    async def start(self) -> None:
        self.logger.info("Starting Telegram bot...")
        await self.bot.start()
        self.me = await self.bot.get_me()
        self.logger.info("Telegram bot started.")
        await self.bot.run_until_disconnected()

    async def _should_respond(self, event) -> bool:
        message_text = event.raw_text or ""
        if message_text.startswith(self.command_prefix):
            return True

        if event.reply_to_msg_id:
            try:
                replied_msg = await event.get_reply_message()
                if replied_msg:
                    await self.__ensure_identity()
                    if self.me and replied_msg.sender_id == self.me.id:
                        return True
            except Exception as exc:
                self.logger.error("Error checking reply message: %s", exc)

        return False

    def _merge_attachment_lists(
        self, *attachment_groups: list[ImageAttachment]
    ) -> list[ImageAttachment]:
        merged: list[ImageAttachment] = []
        seen: set[tuple[str | None, str | None, int]] = set()

        for group in attachment_groups:
            for attachment in group or []:
                signature = (
                    attachment.get("base64"),
                    attachment.get("mime_type"),
                    int(attachment.get("size_bytes", 0)),
                )
                if signature in seen:
                    continue

                copied_attachment = cast(ImageAttachment, dict(attachment))
                merged.append(copied_attachment)
                seen.add(signature)

        return merged

    def _extract_referenced_messages(
        self, history: list[MessagePayload]
    ) -> tuple[list[str], list[ImageAttachment]]:
        context_texts: list[str] = []
        attachments: list[ImageAttachment] = []
        collected: bool = False

        while history:
            candidate = history[-1]
            if candidate["role"] == "assistant":
                break

            if not candidate["attachments"]:
                if collected:
                    break
                # Stop if the most recent messages do not include images.
                break

            context_texts.append(candidate["content"])
            for attachment in candidate["attachments"]:
                copied_attachment = cast(ImageAttachment, dict(attachment))
                attachments.append(copied_attachment)

            history.pop()
            collected = True

        context_texts.reverse()
        attachments.reverse()
        return context_texts, attachments

    async def _collect_image_attachments(
        self, message, *, strict: bool
    ) -> list[ImageAttachment]:
        if not getattr(message, "download_media", None):
            self.logger.debug("Message has no media to download.")
            return []

        media = getattr(message, "media", None)
        has_photo = (
            getattr(message, "photo", None) is not None
            or getattr(media, "photo", None) is not None
        )

        if not media and not has_photo:
            self.logger.debug("Message has no media.")
            return []

        document = getattr(media, "document", None) if media else None
        mime_type = getattr(document, "mime_type", None) if document else None
        file_name: str | None = None

        if not has_photo and document is None:
            self.logger.debug("Media is not a photo or document.")
            return []

        if document and getattr(document, "attributes", None):
            for attribute in document.attributes:
                file_name = getattr(attribute, "file_name", file_name)
                if file_name:
                    break

        if not mime_type and has_photo:
            mime_type = "image/jpeg"

        if mime_type is None:
            if strict:
                raise UnsupportedImageError(None)
            self.logger.info("Skipping attachment with unknown mime type")
            return []

        if not mime_type.startswith("image/"):
            if strict:
                raise UnsupportedImageError(mime_type)
            self.logger.info(
                "Skipping non-image attachment with mime type: %s", mime_type
            )
            return []

        buffer = BytesIO()
        try:
            await message.download_media(file=buffer)
        except Exception as exc:
            if strict:
                raise ImageDownloadError from exc
            self.logger.warning("Failed to download historical attachment: %s", exc)
            return []

        data = buffer.getvalue()
        size_bytes = len(data)

        if size_bytes == 0:
            if strict:
                raise ImageDownloadError("Empty attachment")
            self.logger.warning("Skipping empty attachment from history")
            return []

        if size_bytes > self._MAX_IMAGE_BYTES:
            if strict:
                raise ImageTooLargeError(size_bytes)
            self.logger.warning(
                "Skipping oversized attachment in history (%s bytes)", size_bytes
            )
            return []

        encoded = base64.b64encode(data).decode("ascii")
        attachment: ImageAttachment = {
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "base64": encoded,
            "file_name": file_name,
        }
        self.logger.info(
            "Collected image attachment: %s (%s bytes)", file_name or "unnamed", size_bytes
        )
        return [attachment]

    async def __ensure_identity(self) -> None:
        if self.me is None:
            self.me = await self.bot.get_me()

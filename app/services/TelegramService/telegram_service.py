from __future__ import annotations


import base64
import logging
from io import BytesIO
from typing import Any, Literal, cast

from telethon import TelegramClient, events
from telethon.errors import MessageTooLongError

from app.entities.message import ImageAttachment, MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from app.services.UserService.user_service_interface import UserServiceInterface
from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)

# Telegram message character limit
TELEGRAM_MESSAGE_LIMIT = 4096


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
        chat_repository: ChatRepositoryInterface,
        user_service: UserServiceInterface,
        admin_ids: list[int],
    ) -> None:
        self.logger: logging.Logger = logger
        self.neibot: NeibotServiceInterface = neibot
        self.bot: TelegramClient = telegram_client
        self.me: Any | None = None
        self.command_prefix: str = command_prefix.lower()  # Normalize to lowercase
        self.chat_repository = chat_repository
        self.user_service = user_service
        self.admin_ids = admin_ids

        self.logger.info(
            "TelegramService initialized with command prefix '%s'",
            self.command_prefix,
        )

    @classmethod
    async def create(
        cls,
        command_prefix: str,
        neibot: NeibotServiceInterface,
        telegram_client: TelegramClient,
        logger: logging.Logger,
        chat_repository: ChatRepositoryInterface,
        user_service: UserServiceInterface,
        admin_ids: list[int],
    ) -> TelegramService:
        """Factory to perform async setup steps before returning the service."""
        service = cls(
            command_prefix=command_prefix,
            neibot=neibot,
            telegram_client=telegram_client,
            logger=logger,
            chat_repository=chat_repository,
            user_service=user_service,
            admin_ids=admin_ids,
        )
        service.bot.add_event_handler(service._my_event_handler, events.NewMessage)
        return service

    async def _my_event_handler(self, event) -> None:
        if not await self._should_respond(event):
            self.logger.debug("Ignoring message: %s", event.raw_text)
            return

        await self.__ensure_identity()

        raw_message: str = event.raw_text or ""
        raw_message_stripped = raw_message.strip()
        raw_message_lower = raw_message_stripped.lower()

        # Handle /help command
        if raw_message_lower.startswith("/help") or raw_message_lower.startswith(
            "/start"
        ):
            await self._handle_help_command(event)
            return

        # Handle /save command (manual memory extraction)
        if raw_message_lower.startswith("/save"):
            await self._handle_save_command(event)
            return

        # Handle /memory command (view/manage stored facts)
        if raw_message_lower.startswith("/memory"):
            await self._handle_memory_command(event)
            return

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

        if raw_message_lower.startswith(self.command_prefix):
            user_message = raw_message_stripped[len(self.command_prefix) :].strip()
        else:
            user_message = raw_message_stripped

        metadata: str = await self.__build_metadata(event)

        is_pro = self.user_service.is_user_pro(event.sender_id)
        max_history_turns = self.user_service.get_user_max_history_turns(
            event.sender_id
        )

        # Rolling Window: Only the last 20 messages are sent to the LLM.
        ROLLING_WINDOW_SIZE = 20

        # For Pro users, we only need to fetch the window size.
        # For Free users, we fetch up to the limit to enforce the wall.
        fetch_limit = ROLLING_WINDOW_SIZE if is_pro else max_history_turns

        is_private = await self._is_private_chat(event)

        if is_private:
            # Private Chat: Always use Rolling Window (last N messages)
            # No hard limits, just smooth context sliding.
            history = await self._build_recent_history(event, ROLLING_WINDOW_SIZE)
        else:
            # Group Chat: Explicit threads via replies
            history = await self.__build_reply_history(event, fetch_limit)

        if not is_private and not is_pro and len(history) >= max_history_turns:
            self.logger.warning(
                "Conversation history limit reached (%s). Rejecting request.",
                max_history_turns,
            )

            await event.reply(
                message=f"La conversación ha alcanzado el límite de {max_history_turns} mensajes. Por favor, inicia un nuevo hilo o discusión con el comando {self.command_prefix} para continuar.",
            )
            return

        # Apply rolling window slice for context
        context = history[-ROLLING_WINDOW_SIZE:]

        context_texts, history_attachments = self._extract_referenced_messages(context)
        attachments = self._merge_attachment_lists(history_attachments, attachments)

        user_segment: str = f"{metadata}: {user_message}" if user_message else metadata

        filtered_context: list[str] = [
            text for text in context_texts if text and text != user_segment
        ]

        payload_parts: list[str] = filtered_context + [user_segment]
        payload: str = "\n".join(part for part in payload_parts if part)

        context.append(
            {
                "role": "user",
                "content": payload,
                "attachments": attachments,
            }
        )

        self.logger.info(
            "Constructed payload for Neibot with %s messages in history",
            len(context),
        )

        # Determine model to use based on user tier
        model_name = self.user_service.get_user_model(event.sender_id)

        async with self.bot.action(event.chat_id, "typing"):
            response: str = await self.neibot.get_response(
                context, model_name=model_name, user_id=str(event.sender_id)
            )

        sent_msg = None
        final_response = response  # Track what was actually sent

        try:
            sent_msg = await event.reply(response)
        except MessageTooLongError:
            self.logger.warning(
                "Response too long (%d chars), activating distillation protocol",
                len(response),
            )
            # Activate the distillation protocol to condense the response
            # Pass last 10 messages (5 exchanges) for context
            async with self.bot.action(event.chat_id, "typing"):
                distilled_response = await self.neibot.distill_response(
                    response, context=context[-10:]
                )

            # Defensive handling: distilled_response might still exceed Telegram's limit
            try:
                sent_msg = await event.reply(distilled_response)
                final_response = distilled_response
            except MessageTooLongError:
                self.logger.warning(
                    "Distilled response still too long (%d chars), truncating to %d chars",
                    len(distilled_response),
                    TELEGRAM_MESSAGE_LIMIT,
                )
                # Truncate preserving the end (most important information is usually at the end)
                # Reserve space for truncation notice
                truncation_notice = (
                    "\n\n[... mensaje truncado por límite de caracteres]"
                )
                max_content_length = TELEGRAM_MESSAGE_LIMIT - len(truncation_notice)
                truncated_response = (
                    distilled_response[-max_content_length:] + truncation_notice
                )
                sent_msg = await event.reply(truncated_response)
                final_response = truncated_response
            except Exception as e:
                self.logger.error(
                    "Failed to send distilled response (%d chars): %s",
                    len(distilled_response),
                    str(e),
                    exc_info=True,
                )
                # Send a fallback message to inform the user
                sent_msg = await event.reply(
                    "Lo siento, hubo un error al enviar la respuesta destilada. "
                    "Por favor, intenta reformular tu pregunta de manera más específica."
                )
                final_response = "Error de respuesta"

        # Cache messages for private chats (rolling window)
        if is_private and sent_msg:
            try:
                # Save user message
                user_payload: MessagePayload = {
                    "role": "user",
                    "content": user_message,
                    "attachments": attachments,
                }
                self.chat_repository.save_message(
                    event.chat_id, event.id, user_payload, event.reply_to_msg_id
                )

                # Save bot response
                bot_payload: MessagePayload = {
                    "role": "assistant",
                    "content": final_response,
                    "attachments": [],
                }
                self.chat_repository.save_message(
                    event.chat_id, sent_msg.id, bot_payload, event.id
                )

                self.logger.debug(
                    "Cached user message %s and bot response %s", event.id, sent_msg.id
                )
            except Exception as cache_err:
                self.logger.warning("Failed to cache messages: %s", cache_err)

    async def _build_recent_history(
        self, event, max_count: int
    ) -> list[MessagePayload]:
        """
        Builds history from the most recent messages in the chat (Rolling Window).
        Used for private chats where context is chronological, not thread-based.

        First attempts to read from cache (DB), then falls back to Telegram API.
        Any messages fetched from Telegram are saved to cache.
        """
        history: list[MessagePayload] = []
        chat_id = event.chat_id
        current_msg_id = event.id

        try:
            # Step 1: Try to get cached messages
            cached_messages = self.chat_repository.get_recent_messages(
                chat_id, max_count
            )

            if cached_messages:
                # Filter to only messages BEFORE current message
                cached_messages = [
                    m for m in cached_messages if m["message_id"] < current_msg_id
                ]

                # Reverse to get chronological order (oldest first)
                for cached in reversed(cached_messages[:max_count]):
                    history.append(cached["payload"])

                self.logger.info(
                    "Built recent history with %s cached messages", len(history)
                )
                return history

            # Step 2: Fallback to Telegram API if cache is empty
            messages = await self.bot.get_messages(
                chat_id, limit=max_count, max_id=current_msg_id
            )

            if not messages:
                return history

            await self.__ensure_identity()

            # Process in reverse order (oldest first)
            for msg in reversed(messages):
                if not msg.raw_text and not msg.media:
                    continue

                role: Literal["assistant", "user"] = (
                    "assistant" if self.me and msg.sender_id == self.me.id else "user"
                )

                message_content = msg.raw_text or ""

                attachments = await self._collect_image_attachments(msg, strict=False)

                payload: MessagePayload = {
                    "role": role,
                    "content": message_content,
                    "attachments": attachments,
                }
                history.append(payload)

                # Save to cache for future requests
                self.chat_repository.save_message(
                    chat_id, msg.id, payload, msg.reply_to_msg_id
                )

            self.logger.info(
                "Built recent history with %s messages from Telegram (now cached)",
                len(history),
            )

        except Exception as exc:
            self.logger.error("Error fetching recent history: %s", exc)

        return history

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

    async def __build_reply_history(
        self, event, max_history_turns: int
    ) -> list[MessagePayload]:
        history: list[MessagePayload] = []
        current_msg_id = event.reply_to_msg_id
        chat_id = event.chat_id

        while current_msg_id:
            if len(history) >= max_history_turns:
                self.logger.warning(
                    "Reached max history turns (%s). Stopping fetch.",
                    max_history_turns,
                )
                break

            try:
                cached = self.chat_repository.get_message(chat_id, current_msg_id)
                if cached:
                    history.insert(0, cached["payload"])
                    current_msg_id = cached["reply_to_msg_id"]
                    self.logger.debug("Loaded message %s from DB", current_msg_id)
                    continue

                replied_msg = await self.bot.get_messages(chat_id, ids=current_msg_id)
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

                payload: MessagePayload = {
                    "role": role,
                    "content": message_content,
                    "attachments": attachments,
                }

                history.insert(0, payload)

                next_reply_id = replied_msg.reply_to_msg_id
                self.chat_repository.save_message(
                    chat_id, current_msg_id, payload, next_reply_id
                )

                current_msg_id = next_reply_id

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

    async def _is_private_chat(self, event) -> bool:
        """Check if message is from a private/DM chat.

        Uses Telethon's built-in is_private property from ChatGetter,
        which reliably identifies direct messages vs groups/channels.
        """
        try:
            return event.is_private
        except Exception:
            return False

    async def _should_respond(self, event) -> bool:
        message_text = event.raw_text or ""
        message_text = message_text.strip().lower()

        # 1. Always respond to commands
        if message_text.startswith(self.command_prefix):
            return True
        elif message_text.startswith("/help") or message_text.startswith("/start"):
            return True
        elif message_text.startswith("/save") or message_text.startswith("/memory"):
            return True
        elif "@yunoaidotcom" in message_text:
            return True

        # 2. Private Chat: Respond to EVERYTHING (Rolling Window)
        if await self._is_private_chat(event):
            return True

        # 3. Group Chat: Respond only to replies to bot
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
            "Collected image attachment: %s (%s bytes)",
            file_name or "unnamed",
            size_bytes,
        )
        return [attachment]

    async def __ensure_identity(self) -> None:
        if self.me is None:
            self.me = await self.bot.get_me()

    async def _handle_help_command(self, event) -> None:
        """Sends a help message explaining how to use the bot."""
        if await self._is_private_chat(event):
            help_message = (
                "¡Hola! Soy Yuno. 🧠\n\n"
                "**En este chat privado, funciono en modo continuo.**\n"
                "Simplemente escribe tus mensajes y yo te responderé manteniendo el contexto "
                "de los últimos mensajes.\n\n"
                "No hace falta usar comandos ni responder (reply) a mensajes específicos, "
                "solo conversa fluyéndamente."
            )
        else:
            help_message = (
                "¡Hola! Soy Yuno. 🧠\n\n"
                "Funciono mediante **hilos de conversación** (threads). "
                "Para mantener el contexto y continuar nuestra charla, es necesario que "
                "**respondas a mi último mensaje** (Reply).\n\n"
                f"Para iniciar una conversación escribe: {self.command_prefix} seguido de tu mensaje.\n\n"
                "Por ejemplo:\n"
                f"`{self.command_prefix} tengo esta idea y me gustaria que la pongas a prueba`\n\n"
                "O bien mencionando @yunoaidotcom en tu mensaje.\n\n"
            )
        await event.reply(help_message)

    async def _handle_save_command(self, event) -> None:
        """
        Extracts facts from conversation history and saves them to memory.
        """
        try:
            await event.reply(
                "Analizando la conversación para extraer datos importantes..."
            )

            # 1. Fetch history
            max_history = 50  # Analyze last 50 messages for context
            history = await self.__build_reply_history(event, max_history)

            # 2. Extract and save facts using Neibot
            user_id = str(event.sender_id)
            async with self.bot.action(event.chat_id, "typing"):
                saved_count = await self.neibot.capture_facts_from_history(
                    history, user_id=user_id
                )

            # 3. Reply
            if saved_count > 0:
                await event.reply(
                    f"He guardado {saved_count} nuevos datos sobre ti en mi memoria."
                )
            else:
                await event.reply(
                    "No encontré nuevos datos relevantes para guardar o el servicio de memoria no está disponible."
                )

        except Exception as e:
            self.logger.error(f"Error in /save command: {e}", exc_info=True)
            await event.reply("Ocurrió un error al intentar guardar la memoria.")

    async def _handle_memory_command(self, event) -> None:
        """
        Allows users to view and manage their stored memories.

        Usage:
            /memory         - List all stored memories
            /memory clear   - Delete all memories for this user
        """
        try:
            user_id = str(event.sender_id)
            raw_message = (event.raw_text or "").strip().lower()

            if hasattr(self.neibot, "memory_service") and self.neibot.memory_service:
                await self._handle_memory_command_mem0(event, user_id, raw_message)
                return

            await event.reply("El servicio de memoria no está disponible.")

        except Exception as e:
            self.logger.error(f"Error in /memory command: {e}", exc_info=True)
            await event.reply("Ocurrió un error al consultar la memoria.")

    async def _handle_memory_command_mem0(
        self, event, user_id: str, raw_message: str
    ) -> None:
        """Handle /memory using mem0 (new system)."""
        memory_service = self.neibot.memory_service
        if memory_service is None:
            await event.reply("El servicio de memoria no está disponible.")
            return

        # Handle /memory clear - explicit subcommand parsing
        tokens = raw_message.split()
        if len(tokens) >= 2 and tokens[1].strip().lower() == "clear":
            deleted_count = memory_service.delete_all(user_id)
            if deleted_count > 0:
                await event.reply(f"He borrado {deleted_count} memorias sobre ti.")
            else:
                await event.reply("No tienes memorias guardadas.")
            return

        # Default: list memories
        memories = memory_service.get_all(user_id)

        if not memories:
            await event.reply(
                "No tengo memorias guardadas sobre ti aún.\n\n"
                "Usa /save para guardar datos de nuestra conversación."
            )
            return

        # Format memories by category
        by_category: dict[str, list[str]] = {
            "TECH_STACK": [],
            "BUSINESS_LOGIC": [],
            "USER_CONSTRAINTS": [],
            "OTHER": [],
        }

        # Categories are embedded in the text as "[CATEGORY] text"
        from app.services.MemoryService.memory_service import parse_embedded_category

        for mem in memories:
            text = mem.get("memory") or mem.get("text", "")
            category, clean_text = parse_embedded_category(text)
            if category not in by_category:
                category = "OTHER"
            by_category[category].append(f"• {clean_text}")

        # Build response
        response_parts = ["🧠 **Tu memoria en Yuno:**\n"]

        category_labels = {
            "TECH_STACK": "💻 **Stack Técnico**",
            "BUSINESS_LOGIC": "📋 **Lógica de Negocio**",
            "USER_CONSTRAINTS": "⚠️ **Restricciones**",
            "OTHER": "📝 **Otros**",
        }

        for cat, label in category_labels.items():
            if by_category[cat]:
                response_parts.append(f"\n{label}")
                response_parts.extend(by_category[cat])

        response_parts.append("\n\n_Usa /memory clear para borrar todo._")
        await event.reply("\n".join(response_parts))

from typing import Any

import asyncio
import logging
from telethon import TelegramClient, events

from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service_interface import TelegramServiceInterface


class TelegramService(TelegramServiceInterface):
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
    ) -> "TelegramService":
        """Factory to perform async setup steps before returning the service."""
        service = cls(command_prefix=command_prefix, neibot=neibot, telegram_client=telegram_client, logger=logger)
        service.bot.add_event_handler(service._my_event_handler, events.NewMessage)
        return service

    async def _my_event_handler(self, event) -> None:
        # Check if we should respond (either starts with prefix or is replying to our bot)
        should_respond = await self._should_respond(event)
        if not should_respond:
            self.logger.debug("Ignoring message: %s", event.raw_text)
            return
        
        # Remove command prefix and trim unnecessary whitespace
        user_message: str = str(event.raw_text).replace(self.command_prefix, "").strip()

        # Build contextual metadata so Neibot knows who spoke and where
        metadata: str = await self.__build_metadata(event)
        
        # Build conversation history from reply chain (if this is a reply)
        history: list[tuple[str, str]] = await self.__build_reply_history(event)
        
        # Add current message to history
        payload: str = f"{metadata}: {user_message}"
        history.append(("user", payload))
            
        self.logger.info(f"Constructed payload for Neibot with {len(history)} messages in history")

        # Call Neibot service to get a response while showing typing indicator
        async with self.bot.action(event.chat_id, "typing"):
            response: str = await asyncio.to_thread(self.neibot.get_response, history)

        await event.reply(response)

    async def __build_metadata(self, event) -> str:
        chat = await event.get_chat()
        sender = await event.get_sender()
        chat_name: str = getattr(chat, "title", None) or getattr(chat, "username", None) or "Direct"
        sender_name_parts = [getattr(sender, "first_name", ""), getattr(sender, "last_name", "")]
        sender_name_candidates = [getattr(sender, "username", None), " ".join(part for part in sender_name_parts if part).strip()]
        sender_name: str = next((name for name in sender_name_candidates if name), str(event.sender_id))
        return f"[Group: {chat_name}][User: {sender_name}]"

    async def __build_reply_history(self, event) -> list[tuple[str, str]]:
        """
        Build conversation history by following the reply chain backwards.
        Returns a list of (role, message) tuples in chronological order (oldest first).
        """
        history: list[tuple[str, str]] = []
        current_event = event
        
        # Follow the reply chain backwards
        while current_event.reply_to_msg_id:
            try:
                # Get the message being replied to
                replied_msg = await current_event.get_reply_message()
                if not replied_msg:
                    break
                
                # Get sender info for the replied message
                replied_sender = await replied_msg.get_sender()
                replied_sender_name_parts = [
                    getattr(replied_sender, "first_name", ""),
                    getattr(replied_sender, "last_name", "")
                ]
                replied_sender_name_candidates = [
                    getattr(replied_sender, "username", None),
                    " ".join(part for part in replied_sender_name_parts if part).strip()
                ]
                replied_sender_name: str = next(
                    (name for name in replied_sender_name_candidates if name),
                    str(replied_msg.sender_id)
                )
                
                # Determine role: if the message is from our bot, it's "assistant", otherwise "user"
                role: str = "assistant" if replied_msg.sender_id == self.me.id else "user"
                
                # Build metadata for the replied message
                chat = await replied_msg.get_chat()
                chat_name: str = getattr(chat, "title", None) or getattr(chat, "username", None) or "Direct"
                
                # Format the message
                if role == "assistant":
                    # For assistant messages, just use the text without metadata
                    message_content = replied_msg.raw_text
                else:
                    # For user messages, include metadata
                    message_content = f"[Group: {chat_name}][User: {replied_sender_name}]: {replied_msg.raw_text}"
                
                # Insert at the beginning to maintain chronological order
                history.insert(0, (role, message_content))
                
                # Move to the next message in the chain
                current_event = replied_msg
                
                self.logger.debug(f"Added message to history: {role} - {message_content[:50]}...")
                
            except Exception as e:
                self.logger.error(f"Error fetching reply message: {e}")
                break
        
        self.logger.info(f"Built reply history with {len(history)} previous messages")
        return history

    async def start(self) -> None:
        self.logger.info("Starting Telegram bot...")
        await self.bot.start()
        self.me = await self.bot.get_me()
        self.logger.info("Telegram bot started.")
        await self.bot.run_until_disconnected()

    async def _should_respond(self, event) -> bool:
        """
        Determine if the bot should respond to this message.
        Responds if:
        1. Message starts with the command prefix, OR
        2. Message is a reply to one of our bot's messages
        """
        # Check if message starts with command prefix
        message_text = event.raw_text or ""
        if message_text.startswith(self.command_prefix):
            return True
        
        # Check if this is a reply to our bot
        if event.reply_to_msg_id:
            try:
                replied_msg = await event.get_reply_message()
                if replied_msg and replied_msg.sender_id == self.me.id:
                    return True
            except Exception as e:
                self.logger.error(f"Error checking reply message: {e}")
        
        return False

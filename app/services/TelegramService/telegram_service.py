from typing import Any

import logging
from telethon import TelegramClient, events

from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service_interface import TelegramServiceInterface


class TelegramService(TelegramServiceInterface):
    def __init__(
        self,
        neibot: NeibotServiceInterface,
        telegram_client: TelegramClient,
        logger: logging.Logger,
    ) -> None:
        self.logger: logging.Logger = logger
        self.neibot: NeibotServiceInterface = neibot
        self.bot: TelegramClient = telegram_client
        self.me: Any | None = None

    @classmethod
    async def create(
        cls,
        neibot: NeibotServiceInterface,
        telegram_client: TelegramClient,
        logger: logging.Logger,
    ) -> "TelegramService":
        """Factory to perform async setup steps before returning the service."""
        service = cls(neibot=neibot, telegram_client=telegram_client, logger=logger)
        service.bot.add_event_handler(service._my_event_handler, events.NewMessage)
        return service

    async def _my_event_handler(self, event) -> None:
        if "hello" in event.raw_text and self._should_respond(event.raw_text):
            self.logger.info(
                "Received message: %s from %s", event.raw_text, event.sender_id
            )
            await event.reply("world")

    async def start(self) -> None:
        self.logger.info("Starting Telegram bot...")
        await self.bot.start()
        self.me = await self.bot.get_me()
        self.logger.info("Telegram bot started.")
        await self.bot.run_until_disconnected()

    def _should_respond(self, message: str) -> bool:
        return True

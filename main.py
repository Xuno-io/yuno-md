import asyncio
from app.bootstrap.bootstrapper import bootstrap_bot
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)


async def main():
    bot: TelegramServiceInterface = await bootstrap_bot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())

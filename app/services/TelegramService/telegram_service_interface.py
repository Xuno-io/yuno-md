from abc import ABC, abstractmethod


class TelegramServiceInterface(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Begin processing Telegram events."""
        raise NotImplementedError

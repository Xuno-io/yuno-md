from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.neibot_service import NeibotService
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import TelegramServiceInterface
from openai import OpenAI
from xuno_components.logger.logger_interface import LoggerInterface
from telethon import TelegramClient

def get_neibot_service(components: Components) -> NeibotServiceInterface:
    return NeibotService(
        openai_client=components.get_component(OpenAI),
        logger=components.get_component(LoggerInterface).get_logger('NeibotService')
    )

async def get_telegram_service(
    components: Components, neibot_service: NeibotServiceInterface
) -> TelegramServiceInterface:
    return await TelegramService.create(
        neibot=neibot_service,
        telegram_client=components.get_component(TelegramClient),
        logger=components.get_component(LoggerInterface).get_logger("TelegramService"),
    )
    

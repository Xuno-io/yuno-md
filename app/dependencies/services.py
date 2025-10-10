from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.neibot_service import NeibotService
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import TelegramServiceInterface
from openai import OpenAI
from xuno_components.logger.logger_interface import LoggerInterface
from telethon import TelegramClient
from xuno_components.configuration.configuration_interface import ConfigurationInterface

def get_neibot_service(components: Components) -> NeibotServiceInterface:
    model_name: str = components.get_component(ConfigurationInterface).get_configuration('MODEL_NAME', str)
    system_prompt: str = components.get_component(ConfigurationInterface).get_configuration('SYSTEM_PROMPT', str)
    
    return NeibotService(
        system_prompt=system_prompt,
        model_name=model_name,
        openai_client=components.get_component(OpenAI),
        logger=components.get_component(LoggerInterface).get_logger('NeibotService')
    )

async def get_telegram_service(
    components: Components, neibot_service: NeibotServiceInterface
) -> TelegramServiceInterface:
    return await TelegramService.create(
        command_prefix="/yuno",
        neibot=neibot_service,
        telegram_client=components.get_component(TelegramClient),
        logger=components.get_component(LoggerInterface).get_logger("TelegramService"),
    )
    

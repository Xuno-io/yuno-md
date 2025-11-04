import os

from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service import NeibotService
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from openai import OpenAI
from telethon import TelegramClient
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger_interface import LoggerInterface


def get_neibot_service(components: Components) -> NeibotServiceInterface:
    configuration = components.get_component(ConfigurationInterface)
    model_name: str = configuration.get_configuration("MODEL_NAME", str)
    system_prompt: str = configuration.get_configuration("SYSTEM_PROMPT", str)

    creator_username = os.getenv("CREATOR_USERNAME", "").strip()
    if not creator_username:
        raise ValueError(
            "Environment variable CREATOR_USERNAME must be set with a valid username."
        )

    if not creator_username.startswith("@"):
        creator_username = f"@{creator_username}"

    system_prompt = system_prompt.replace("{CREATOR_USERNAME}", creator_username)

    return NeibotService(
        system_prompt=system_prompt,
        model_name=model_name,
        openai_client=components.get_component(OpenAI),
        logger=components.get_component(LoggerInterface).get_logger("NeibotService"),
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

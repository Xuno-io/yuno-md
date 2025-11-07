import os

from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service import NeibotService
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.neibot_dspy_service import NeibotDSPyService
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from openai import OpenAI
from telethon import TelegramClient
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger_interface import LoggerInterface
import dspy


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


def get_neibot_dspy_service(components: Components) -> NeibotServiceInterface:
    """
    Create a DSPy-powered Neibot service with YunoAI and Langfuse tracing.

    This service uses DSPy for structured LLM programming and Langfuse for observability.
    """
    lm: dspy.LM = components.get_component(dspy.LM)
    configuration = components.get_component(ConfigurationInterface)
    system_prompt = configuration.get_configuration("SYSTEM_PROMPT", str)

    # Get creator username
    creator_username = os.getenv("CREATOR_USERNAME", "").strip()
    if not creator_username:
        raise ValueError(
            "Environment variable CREATOR_USERNAME must be set with a valid username."
        )

    if not creator_username.startswith("@"):
        creator_username = f"@{creator_username}"

    system_prompt = system_prompt.replace("{CREATOR_USERNAME}", creator_username)

    return NeibotDSPyService(
        system_prompt=system_prompt,
        lm=lm,
        logger=components.get_component(LoggerInterface).get_logger(
            "NeibotDSPyService"
        ),
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

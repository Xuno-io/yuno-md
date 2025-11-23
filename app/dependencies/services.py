import os

from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.neibot_dspy_service import NeibotDSPyService
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from app.dependencies.repositories import get_chat_repository
from telethon import TelegramClient
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger_interface import LoggerInterface


def get_neibot_dspy_service(components: Components) -> NeibotServiceInterface:
    """
    Create a DSPy-powered Neibot service with YunoAI and Langfuse tracing.

    This service uses DSPy for structured LLM programming and Langfuse for observability.
    """
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

    # Extract configuration values for LM creation
    model_name = configuration.get_configuration("MODEL_NAME", str)
    openai_endpoint = configuration.get_configuration("OPENAI_ENDPOINT", str)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    temp_val = configuration.get_configuration("DSPY_TEMPERATURE", float, default=0.7)
    temperature = float(temp_val if temp_val is not None else 0.7)

    max_tokens_val = configuration.get_configuration(
        "DSPY_MAX_TOKENS", int, default=8192
    )
    max_tokens = int(max_tokens_val if max_tokens_val is not None else 8192)

    return NeibotDSPyService(
        system_prompt=system_prompt,
        model_name=model_name,
        api_key=openai_api_key,
        api_base=openai_endpoint,
        temperature=temperature,
        max_tokens=max_tokens,
        logger=components.get_component(LoggerInterface).get_logger(
            "NeibotDSPyService"
        ),
    )


async def get_telegram_service(
    components: Components, neibot_service: NeibotServiceInterface
) -> TelegramServiceInterface:
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    admin_ids = [
        int(id.strip()) for id in admin_ids_str.split(",") if id.strip().isdigit()
    ]

    configuration = components.get_component(ConfigurationInterface)
    max_history_turns = configuration.get_configuration(
        "MAX_HISTORY_TURNS", int, default=100
    )

    return await TelegramService.create(
        command_prefix="/yuno",
        neibot=neibot_service,
        telegram_client=components.get_component(TelegramClient),
        logger=components.get_component(LoggerInterface).get_logger("TelegramService"),
        chat_repository=get_chat_repository(components),
        admin_ids=admin_ids,
        max_history_turns=(
            int(max_history_turns) if max_history_turns is not None else 50
        ),
    )

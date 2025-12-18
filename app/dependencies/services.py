import os
from pathlib import Path

from app.bootstrap.components import Components
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.neibot_service import NeibotService

from app.services.MemoryService.memory_service_interface import MemoryServiceInterface
from app.services.MemoryService.memory_service import MemoryService
from app.services.TelegramService.telegram_service import TelegramService
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from app.services.UserService.user_service import UserService
from app.services.UserService.user_service_interface import UserServiceInterface
from app.dependencies.repositories import get_chat_repository, get_user_repository
from telethon import TelegramClient
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger_interface import LoggerInterface


def get_neibot_service(components: Components) -> NeibotServiceInterface:
    """
    Create a Neibot service with direct OpenAI/LiteLLM calls and Langfuse tracing.
    """
    configuration = components.get_component(ConfigurationInterface)

    try:
        system_prompt = Path("yuno.prompt").read_text(encoding="utf-8")
    except Exception:
        system_prompt = configuration.get_configuration("SYSTEM_PROMPT", str)

    # Get creator username
    creator_username = os.getenv("CREATOR_USERNAME", "").strip()
    if not creator_username:
        raise ValueError(
            "Environment variable CREATOR_USERNAME must be set with a valid username."
        )

    if not creator_username.startswith("@"):
        creator_username = f"@{creator_username}"

    if "{CREATOR_USERNAME}" in system_prompt:
        system_prompt = system_prompt.replace("{CREATOR_USERNAME}", creator_username)

    # Extract configuration values for LM creation
    model_name = configuration.get_configuration("MODEL_NAME", str)
    location = configuration.get_configuration(
        "VERTEX_LOCATION", str, default="us-central1"
    )
    project_id = os.getenv("VERTEX_PROJECT_ID", "").strip()

    temp_val = configuration.get_configuration("LLM_TEMPERATURE", float, default=0.7)
    temperature = float(temp_val if temp_val is not None else 0.7)

    max_tokens_val = configuration.get_configuration(
        "LLM_MAX_TOKENS", int, default=8192
    )
    max_tokens = int(max_tokens_val if max_tokens_val is not None else 8192)

    cache_threshold_val = configuration.get_configuration(
        "CACHE_TOKEN_THRESHOLD", int, default=2048
    )
    cache_threshold = int(
        cache_threshold_val if cache_threshold_val is not None else 2048
    )

    # Model for memory extraction (fast, cheap, structured output)
    extraction_model = configuration.get_configuration(
        "EXTRACTION_MODEL_NAME", str, default="gemini-3-flash-preview"
    )

    return NeibotService(
        system_prompt=system_prompt,
        model_name=model_name,
        location=location,
        project_id=project_id or None,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_threshold=cache_threshold,
        logger=components.get_component(LoggerInterface).get_logger("NeibotService"),
        memory_service=get_memory_service(components),
        extraction_model_name=extraction_model,
    )


def get_memory_service(components: Components) -> MemoryServiceInterface:
    """
    Create the mem0-based memory service with Redis backend.

    Environment variables:
        REDIS_HOST: Redis host (default: localhost)
        REDIS_PORT: Redis port (default: 6379)
        GOOGLE_API_KEY: Required for embeddings
    """
    configuration = components.get_component(ConfigurationInterface)
    logger = components.get_component(LoggerInterface)

    redis_host = configuration.get_configuration("REDIS_HOST", str, default="localhost")
    redis_port = configuration.get_configuration("REDIS_PORT", int, default=6379)

    # Model for memory operations
    memory_llm_model = configuration.get_configuration(
        "MEMORY_LLM_MODEL", str, default="gemini-2.0-flash"
    )
    memory_embedder_model = configuration.get_configuration(
        "MEMORY_EMBEDDER_MODEL", str, default="text-embedding-004"
    )

    return MemoryService(
        logger=logger.get_logger("MemoryService"),
        redis_host=redis_host,
        redis_port=redis_port,
        llm_model=memory_llm_model,
        embedder_model=memory_embedder_model,
    )


def get_user_service(components: Components) -> UserServiceInterface:
    configuration = components.get_component(ConfigurationInterface)
    default_max_history_turns = configuration.get_configuration(
        "MAX_HISTORY_TURNS", int, default=100
    )
    pro_max_history_turns = configuration.get_configuration(
        "MAX_HISTORY_TURNS_PRO", int, default=200
    )
    default_model_name = configuration.get_configuration(
        "MODEL_NAME", str, default="gemini-3-flash-preview"
    )
    pro_model_name = configuration.get_configuration(
        "MODEL_NAME_PRO", str, default="gemini-3-pro-preview"
    )
    return UserService(
        user_repository=get_user_repository(components),
        default_max_history_turns=default_max_history_turns,
        pro_max_history_turns=pro_max_history_turns,
        default_model_name=default_model_name,
        pro_model_name=pro_model_name,
    )


async def get_telegram_service(
    components: Components, neibot_service: NeibotServiceInterface
) -> TelegramServiceInterface:
    admin_ids_str = os.getenv("ADMIN_IDS", "")
    admin_ids = [
        int(id.strip()) for id in admin_ids_str.split(",") if id.strip().isdigit()
    ]

    configuration = components.get_component(ConfigurationInterface)

    prefix: str = configuration.get_configuration(
        "COMMAND_PREFIX", str, default="/yuno"
    )

    return await TelegramService.create(
        command_prefix=prefix,
        neibot=neibot_service,
        telegram_client=components.get_component(TelegramClient),
        logger=components.get_component(LoggerInterface).get_logger("TelegramService"),
        chat_repository=get_chat_repository(components),
        user_service=get_user_service(components),
        admin_ids=admin_ids,
    )

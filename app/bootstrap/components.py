import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import httpx

from xuno_components.cache.cache_interface import CacheInterface
from xuno_components.cache.redis_cache import RedisCache
from xuno_components.configuration.configuration import Configuration
from xuno_components.configuration.configuration_interface import \
    ConfigurationInterface
from xuno_components.logger.logger import Logger
from xuno_components.logger.logger_interface import LoggerInterface
from openai import OpenAI
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

class ComponentsMeta(type):
    _instances: Dict = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        env = args[0] if args else kwargs.get("env")
        key = (cls, env)
        with cls._lock:
            if key not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[key] = instance
        return cls._instances[key]


class Components(metaclass=ComponentsMeta):

    def __init__(self, env: str, config_path: str) -> None:
        self.__env: str = env
        root_dir: str = str(Path(__file__).resolve().parents[2])
        self.__config_path: str = os.path.join(root_dir, config_path)
        self.__components: Dict[str, Any] = self.__bootstrap_components()

    def __bootstrap_components(self) -> Dict[str, Any]:
        if self.__env == 'development':
            return self.__get_dev_components()

        raise ValueError(f'Invalid environment: {self.__env}')

    def __get_dev_components(self) -> Dict[Any, Any]:
        configuration: ConfigurationInterface = Configuration(
            self.__env, self.__config_path)

        logger: LoggerInterface = Logger(
            log_format=configuration.get_configuration('LOG_FORMAT', str),
            log_level=configuration.get_configuration('LOG_LEVEL', str),
        )
        
        _logger_instance = logger.get_logger('Components')

        # Configure OpenAI client with timeout and retries from configuration
        openai_endpoint: str = configuration.get_configuration(
            'OPENAI_ENDPOINT', str)
        timeout_seconds: float = float(configuration.get_configuration(
            'OPENAI_TIMEOUT_SECONDS', float, default=30.0) or 30.0)  # type: ignore[arg-type]
        max_retries: int = int(configuration.get_configuration(
            'OPENAI_MAX_RETRIES', int, default=2) or 2)
        try:
            # Prefer granular timeout config
            timeout: httpx.Timeout = httpx.Timeout(timeout_seconds)
        except Exception:
            timeout = timeout_seconds  # type: ignore[assignment]
        
        _logger_instance.info(
            f"OpenAI client configured with endpoint: {openai_endpoint}, "
            f"timeout: {timeout}, max_retries: {max_retries}"
        )
        
        openai_client: OpenAI = OpenAI(
            base_url=openai_endpoint,
            timeout=timeout, max_retries=max_retries)
        # Conditional MCP setup based on configuration flag
        # Redis client timeouts/retry - expose via configuration with safe defaults
        redis_socket_connect_timeout: float = float(configuration.get_configuration(
            'REDIS_SOCKET_CONNECT_TIMEOUT', float, default=2.0) or 2.0)  # type: ignore[arg-type]
        redis_socket_timeout: float = float(configuration.get_configuration(
            'REDIS_SOCKET_TIMEOUT', float, default=5.0) or 5.0)  # type: ignore[arg-type]
        redis_retry_on_timeout: bool = bool(configuration.get_configuration(
            'REDIS_RETRY_ON_TIMEOUT', bool, default=True) or True)

        redis_cache: CacheInterface = RedisCache(
            host=configuration.get_configuration('REDIS_HOST', str),
            port=configuration.get_configuration('REDIS_PORT', int),
            db=0,  # Using db 0 for conversation history, mem0 uses default
            # Pass explicit client-level socket timeouts & retry_on_timeout
            socket_connect_timeout=redis_socket_connect_timeout,
            socket_timeout=redis_socket_timeout,
            retry_on_timeout=redis_retry_on_timeout,
        )
        
        # Telegram client setup
        telegram_api_id: int = configuration.get_configuration('TELEGRAM_API_ID', int)
        telegram_api_hash: str = configuration.get_configuration('TELEGRAM_APP_HASH', str)
        
        telegram_client: TelegramClient = TelegramClient(
            "bot",
            telegram_api_id,
            telegram_api_hash
        )

        components: Dict[Any, Any] = {
            ConfigurationInterface: configuration,
            OpenAI: openai_client,
            LoggerInterface: logger,
            CacheInterface: redis_cache,
            TelegramClient: telegram_client,
        }

        return components

    def get_component(self, component_name: Any) -> Any:
        if component_name not in self.__components:
            raise ValueError(f'Component {component_name} not found')

        return self.__components[component_name]

    def get_config_path(self) -> str:
        return self.__config_path

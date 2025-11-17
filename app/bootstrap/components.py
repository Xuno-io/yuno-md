import os
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar, cast

import httpx

from xuno_components.cache.cache_interface import CacheInterface
from xuno_components.configuration.configuration import Configuration
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger import Logger
from xuno_components.logger.logger_interface import LoggerInterface

# from openai import OpenAI
from telethon import TelegramClient
from dotenv import load_dotenv
from langfuse.openai import OpenAI

# DSPy Langfuse instrumentation
from openinference.instrumentation.dspy import DSPyInstrumentor
import dspy

load_dotenv()


def _is_test_environment() -> bool:
    """
    Check if we are running in a test environment.

    Returns:
        True if running under pytest or if TESTING env var is set, False otherwise.
    """
    import sys
    
    # Check if pytest is in the command line arguments
    if any("pytest" in arg for arg in sys.argv):
        return True

    # Check for TESTING environment variable
    if os.getenv("TESTING", "").lower() in ("true", "1", "yes"):
        return True

    return False


def _validate_otel_env_vars() -> None:
    """
    Validate OpenTelemetry/Langfuse environment variables required for DSPy instrumentation.

    Checks for OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS.
    If headers are not provided directly, attempts to build them from LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY using Basic authentication.

    Raises:
        RuntimeError: If required environment variables are missing or empty.
    """
    import base64

    otel_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    otel_headers: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").strip()

    # Optionally build headers from Langfuse keys if not provided directly
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    langfuse_base_url: str = os.getenv("LANGFUSE_BASE_URL").strip()
    
    if langfuse_public_key and langfuse_secret_key and langfuse_base_url:
        return # Skip validation if Langfuse integration is used

    if not otel_endpoint:
        raise RuntimeError(
            "OTEL_EXPORTER_OTLP_ENDPOINT environment variable is not set or is empty. "
            "Please set the OTEL_EXPORTER_OTLP_ENDPOINT environment variable with a valid OTLP endpoint URL."
        )

    if not otel_headers:
        raise RuntimeError(
            "OTEL_EXPORTER_OTLP_HEADERS environment variable is not set or is empty, "
            "and LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not available to build headers. "
            "Please set either OTEL_EXPORTER_OTLP_HEADERS directly (e.g., 'Authorization=Basic <base64_credentials>') "
            "or provide LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to build the headers automatically."
        )


# Validate OpenTelemetry/Langfuse environment variables before instrumentation
# Skip validation and instrumentation in test environment
if not _is_test_environment():
    _validate_otel_env_vars()
    # Initialize DSPy instrumentation for Langfuse tracing
    # This must happen before any DSPy modules are created
    DSPyInstrumentor().instrument()


T = TypeVar("T")


class ComponentsMeta(type):
    _instances: dict[tuple[type, str], "Components"] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        env = args[0] if args else kwargs.get("env")
        if env is None:
            raise ValueError("Environment must be provided")

        env_key = str(env)
        key = (cls, env_key)
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
        self.__components: dict[type[Any], Any] = self.__bootstrap_components()

    def __bootstrap_components(self) -> dict[type[Any], Any]:
        if self.__env in {"development", "staging", "production"}:
            return self.__get_dev_components()

        raise ValueError(f"Invalid environment: {self.__env}")

    def __get_dev_components(self) -> dict[type[Any], Any]:
        configuration: ConfigurationInterface = Configuration(
            self.__env, self.__config_path
        )

        logger: LoggerInterface = Logger(
            log_format=configuration.get_configuration("LOG_FORMAT", str),
            log_level=configuration.get_configuration("LOG_LEVEL", str),
        )
        _logger_instance = logger.get_logger("Components")

        # Configure OpenAI client with timeout and retries from configuration
        openai_endpoint: str = configuration.get_configuration("OPENAI_ENDPOINT", str)
        timeout_seconds: float = float(
            configuration.get_configuration(
                "OPENAI_TIMEOUT_SECONDS", float, default=300.0
            )
            or 300.0
        )  # type: ignore[arg-type]
        max_retries: int = int(
            configuration.get_configuration("OPENAI_MAX_RETRIES", int, default=2) or 2
        )
        try:
            # Prefer granular timeout config
            timeout: httpx.Timeout = httpx.Timeout(timeout_seconds)
        except Exception:
            timeout = timeout_seconds  # type: ignore[assignment]
        # Retrieve and validate OpenAI API key before creating clients
        openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set or is empty. "
                "Please set the OPENAI_API_KEY environment variable with a valid API key."
            )

        _logger_instance.info(
            f"OpenAI client configured with endpoint: {openai_endpoint}, "
            f"timeout: {timeout}, max_retries: {max_retries}"
        )

        openai_client: OpenAI = OpenAI(
            base_url=openai_endpoint, timeout=timeout, max_retries=max_retries
        )
        # Conditional MCP setup based on configuration flag
        # Redis client timeouts/retry - expose via configuration with safe defaults
        redis_socket_connect_timeout: float = float(
            configuration.get_configuration(
                "REDIS_SOCKET_CONNECT_TIMEOUT", float, default=2.0
            )
            or 2.0
        )  # type: ignore[arg-type]
        redis_socket_timeout: float = float(
            configuration.get_configuration("REDIS_SOCKET_TIMEOUT", float, default=5.0)
            or 5.0
        )  # type: ignore[arg-type]
        redis_retry_on_timeout: bool = bool(
            configuration.get_configuration(
                "REDIS_RETRY_ON_TIMEOUT", bool, default=True
            )
            or True
        )

        # Retrieve configuration values and handle None explicitly to preserve zero values
        temp_val = configuration.get_configuration(
            "DSPY_TEMPERATURE", float, default=0.7
        )
        temperature = float(temp_val if temp_val is not None else 0.7)

        max_tokens_val = configuration.get_configuration(
            "DSPY_MAX_TOKENS", int, default=8192
        )
        max_tokens = int(max_tokens_val if max_tokens_val is not None else 8192)

        lm: dspy.LM = dspy.LM(
            model="openai/" + configuration.get_configuration("MODEL_NAME", str),
            api_base=configuration.get_configuration("OPENAI_ENDPOINT", str),
            api_key=openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        """
        redis_cache: CacheInterface = RedisCache(
            host=configuration.get_configuration("REDIS_HOST", str),
            port=configuration.get_configuration("REDIS_PORT", int),
            db=0,  # Using db 0 for conversation history, mem0 uses default
            # Pass explicit client-level socket timeouts & retry_on_timeout
            socket_connect_timeout=redis_socket_connect_timeout,
            socket_timeout=redis_socket_timeout,
            retry_on_timeout=redis_retry_on_timeout,
        )
        """

        # Telegram client setup
        telegram_api_id: int = configuration.get_configuration("TELEGRAM_API_ID", int)
        telegram_api_hash: str = configuration.get_configuration(
            "TELEGRAM_APP_HASH", str
        )

        telegram_client: TelegramClient = TelegramClient(
            "bot", telegram_api_id, telegram_api_hash
        )

        components: dict[type[Any], Any] = {
            ConfigurationInterface: configuration,
            OpenAI: openai_client,
            LoggerInterface: logger,
            CacheInterface: None,  # redis_cache (temporarily disabled),
            TelegramClient: telegram_client,
            dspy.LM: lm,
        }

        return components

    def get_component(self, component_name: type[T]) -> T:
        if component_name not in self.__components:
            raise ValueError(f"Component {component_name} not found")

        return cast(T, self.__components[component_name])

    def get_config_path(self) -> str:
        return self.__config_path

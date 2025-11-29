import os
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar, cast


from xuno_components.configuration.configuration import Configuration
from xuno_components.configuration.configuration_interface import ConfigurationInterface
from xuno_components.logger.logger import Logger
from xuno_components.logger.logger_interface import LoggerInterface
from xuno_components.database.db_interface import DBInterface
from app.components.database.sqlite_db import SqliteDB
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

from telethon import TelegramClient
from dotenv import load_dotenv


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
    Validate OpenTelemetry/Langfuse environment variables for instrumentation.

    This function checks for the necessary environment variables to configure
    OpenTelemetry for Langfuse tracing. It supports two configuration paths:


    1.  **Langfuse Native Integration:** If `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
        and `LANGFUSE_BASE_URL` are all set, validation is skipped, assuming
        a direct Langfuse integration is being used.

    2.  **Manual OpenTelemetry Configuration:** If the Langfuse variables are not
        fully provided, the function requires `OTEL_EXPORTER_OTLP_ENDPOINT` and
        `OTEL_EXPORTER_OTLP_HEADERS` to be set for manual OTLP export.

    Raises:
        RuntimeError: If the required environment variables for manual OpenTelemetry
                      configuration are missing or empty when Langfuse native
                      integration variables are not provided.
    """

    otel_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    otel_headers: str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").strip()

    # Optionally build headers from Langfuse keys if not provided directly
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    langfuse_base_url: str = os.getenv("LANGFUSE_BASE_URL", "").strip()

    if langfuse_public_key and langfuse_secret_key and langfuse_base_url:
        return  # Skip validation if Langfuse integration is used

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

    GoogleGenAIInstrumentor().instrument()

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

        # Telegram client setup
        telegram_api_id: int = configuration.get_configuration("TELEGRAM_API_ID", int)
        telegram_api_hash: str = configuration.get_configuration(
            "TELEGRAM_APP_HASH", str
        )

        telegram_client: TelegramClient = TelegramClient(
            "bot", telegram_api_id, telegram_api_hash
        )

        # Database
        sqlite_db: DBInterface = SqliteDB(
            db_path=configuration.get_configuration("SQLITE_DB_PATH", str)
        )

        sqlite_db.connect()

        components: dict[type[Any], Any] = {
            ConfigurationInterface: configuration,
            LoggerInterface: logger,
            TelegramClient: telegram_client,
            DBInterface: sqlite_db,
        }

        return components

    def get_component(self, component_name: type[T]) -> T:
        if component_name not in self.__components:
            raise ValueError(f"Component {component_name} not found")

        return cast(T, self.__components[component_name])

    def get_config_path(self) -> str:
        return self.__config_path

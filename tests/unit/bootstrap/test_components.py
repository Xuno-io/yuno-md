import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.bootstrap import components
from app.bootstrap.components import Components


@pytest.mark.unit
class TestComponentsOpenAIAPIKeyValidation:
    """Test suite for OpenAI API key validation in Components initialization."""

    def _create_config_directory(self, tmp_path: Path) -> Path:
        """Create a temporary configuration directory with required JSON files."""
        config_dir = tmp_path / "configuration"
        config_dir.mkdir()

        # Create default.json
        default_config = {
            "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "LOG_LEVEL": "INFO",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": 6379,
            "OPENAI_ENDPOINT": "https://api.openai.com/v1",
            "MODEL_NAME": "gpt-4",
            "DSPY_TEMPERATURE": 0.7,
            "DSPY_MAX_TOKENS": 8192,
        }
        (config_dir / "default.json").write_text(json.dumps(default_config, indent=2))

        # Create development.json
        dev_config = {
            "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "LOG_LEVEL": "INFO",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": 6379,
            "OPENAI_ENDPOINT": "https://api.openai.com/v1",
            "MODEL_NAME": "gpt-4",
            "TELEGRAM_API_ID": 123,
            "TELEGRAM_APP_HASH": "test_hash",
            "DSPY_TEMPERATURE": 0.7,
            "DSPY_MAX_TOKENS": 8192,
        }
        (config_dir / "development.json").write_text(json.dumps(dev_config, indent=2))

        return config_dir

    @patch.dict(os.environ, {}, clear=False)
    def test_components_raises_runtime_error_when_openai_api_key_not_set(
        self, tmp_path
    ):
        """Test that Components raises RuntimeError when OPENAI_API_KEY is not set."""
        config_dir = self._create_config_directory(tmp_path)

        # Ensure OPENAI_API_KEY is not set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(RuntimeError) as exc_info:
            Components(env="development", config_path=str(config_dir))

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)
        assert "Please set the OPENAI_API_KEY environment variable" in str(
            exc_info.value
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_components_raises_runtime_error_when_openai_api_key_is_empty(
        self, tmp_path
    ):
        """Test that Components raises RuntimeError when OPENAI_API_KEY is empty."""
        config_dir = self._create_config_directory(tmp_path)

        os.environ["OPENAI_API_KEY"] = ""

        with pytest.raises(RuntimeError) as exc_info:
            Components(env="development", config_path=str(config_dir))

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "   "}, clear=False)
    def test_components_raises_runtime_error_when_openai_api_key_is_whitespace_only(
        self, tmp_path
    ):
        """Test that Components raises RuntimeError when OPENAI_API_KEY is whitespace only."""
        config_dir = self._create_config_directory(tmp_path)

        os.environ["OPENAI_API_KEY"] = "   "

        with pytest.raises(RuntimeError) as exc_info:
            Components(env="development", config_path=str(config_dir))

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)


@pytest.mark.unit
class TestComponentsOTELEnvVarsValidation:
    """Test suite for OpenTelemetry/Langfuse environment variables validation."""

    def test_otel_validation_raises_error_when_endpoint_not_set(self, monkeypatch):
        """Test that OTEL validation raises RuntimeError when OTEL_EXPORTER_OTLP_ENDPOINT is not set."""
        # Remove OTEL variables
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        # Call validation function directly
        with pytest.raises(RuntimeError) as exc_info:
            components._validate_otel_env_vars()

        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)

    def test_otel_validation_raises_error_when_endpoint_is_empty(self, monkeypatch):
        """Test that OTEL validation raises RuntimeError when OTEL_EXPORTER_OTLP_ENDPOINT is empty."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            components._validate_otel_env_vars()

        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)

    def test_otel_validation_raises_error_when_headers_not_set(self, monkeypatch):
        """Test that OTEL validation raises RuntimeError when headers are not set."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.com/otlp")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            components._validate_otel_env_vars()

        assert "OTEL_EXPORTER_OTLP_HEADERS" in str(exc_info.value)
        assert "not set or is empty" in str(exc_info.value)

    def test_otel_validation_succeeds_with_direct_headers(self, monkeypatch):
        """Test that OTEL validation succeeds when headers are provided directly."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.com/otlp")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic dGVzdA==")

        # Should not raise an error
        components._validate_otel_env_vars()

    def test_otel_validation_succeeds_with_langfuse_keys(self, monkeypatch):
        """Test that OTEL validation succeeds when Langfuse keys are provided."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.com/otlp")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        # Should not raise an error
        components._validate_otel_env_vars()

        # Verify that the header was set in the environment
        assert "OTEL_EXPORTER_OTLP_HEADERS" in os.environ
        assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"].startswith(
            "Authorization=Basic "
        )

    def test_otel_validation_prefers_direct_headers_over_keys(self, monkeypatch):
        """Test that OTEL validation prefers direct headers over Langfuse keys."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.com/otlp")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic direct")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        # Should not raise an error and use direct headers
        components._validate_otel_env_vars()

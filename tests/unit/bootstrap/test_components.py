import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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

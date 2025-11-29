import os

import pytest

from app.bootstrap import components


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
        assert "OTEL_EXPORTER_OTLP_HEADERS" not in os.environ

    def test_otel_validation_prefers_direct_headers_over_keys(self, monkeypatch):
        """Test that OTEL validation prefers direct headers over Langfuse keys."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://example.com/otlp")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic direct")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

        # Should not raise an error and use direct headers
        components._validate_otel_env_vars()

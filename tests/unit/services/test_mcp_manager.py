"""
Unit tests for the McpManager module.

Tests cover configuration loading, environment-variable resolution,
error handling, and MCPToolset creation for both Stdio and SSE servers.
"""

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.McpService.mcp_manager import (
    McpManager,
    _resolve_env_vars,
    _resolve_env_in_dict,
    _resolve_env_in_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logger() -> logging.Logger:
    """Return a test logger."""
    return logging.getLogger("test_mcp_manager")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Return a temporary config file path (not yet written)."""
    return tmp_path / "mcp_servers.json"


def _write_config(path: Path, data: dict) -> str:
    """Write *data* as JSON to *path* and return its string representation."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------


class TestResolveEnvVars:
    """Test ${ENV_VAR} placeholder resolution."""

    def test_resolve_single_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _resolve_env_vars("Bearer ${MY_KEY}") == "Bearer secret123"

    def test_resolve_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "8080")
        assert _resolve_env_vars("${HOST}:${PORT}") == "localhost:8080"

    def test_unresolved_var_kept_as_is(self) -> None:
        """Unset env vars should leave the placeholder intact."""
        # Ensure the var does not exist
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        result = _resolve_env_vars("key=${NONEXISTENT_VAR_XYZ}")
        assert result == "key=${NONEXISTENT_VAR_XYZ}"

    def test_no_placeholders(self) -> None:
        assert _resolve_env_vars("plain text") == "plain text"

    def test_resolve_env_in_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOKEN", "abc")
        result = _resolve_env_in_dict({"Authorization": "Bearer ${TOKEN}"})
        assert result == {"Authorization": "Bearer abc"}

    def test_resolve_env_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DIR", "/data")
        result = _resolve_env_in_list(["--path", "${DIR}/files"])
        assert result == ["--path", "/data/files"]


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test JSON configuration file loading."""

    def test_missing_config_file(self, logger: logging.Logger) -> None:
        """Manager should handle a missing config file gracefully."""
        manager = McpManager(config_path="/nonexistent/path.json", logger=logger)
        assert manager.get_toolsets() == []

    def test_invalid_json(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """Manager should handle malformed JSON gracefully."""
        tmp_config.write_text("{invalid json", encoding="utf-8")
        manager = McpManager(config_path=str(tmp_config), logger=logger)
        assert manager.get_toolsets() == []

    def test_empty_servers_list(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """An empty servers list should result in zero toolsets."""
        config_path = _write_config(tmp_config, {"servers": []})
        manager = McpManager(config_path=config_path, logger=logger)
        assert manager.get_toolsets() == []

    def test_no_servers_key(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """A config without 'servers' key should result in zero toolsets."""
        config_path = _write_config(tmp_config, {"version": 1})
        manager = McpManager(config_path=config_path, logger=logger)
        assert manager.get_toolsets() == []


# ---------------------------------------------------------------------------
# Server enable/disable logic
# ---------------------------------------------------------------------------


class TestServerEnableDisable:
    """Test that only enabled servers produce toolsets."""

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_disabled_server_skipped(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "disabled-one",
                        "enabled": False,
                        "type": "stdio",
                        "command": "echo",
                        "args": [],
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert manager.get_toolsets() == []
        mock_toolset_cls.assert_not_called()

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_enabled_server_loaded(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "my-stdio",
                        "enabled": True,
                        "type": "stdio",
                        "command": "echo",
                        "args": ["hello"],
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1
        mock_toolset_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Stdio toolset creation
# ---------------------------------------------------------------------------


class TestStdioToolset:
    """Test Stdio MCP server toolset creation."""

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_stdio_basic(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "fs-server",
                        "enabled": True,
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {"KEY": "val"},
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1

        call_kwargs = mock_toolset_cls.call_args[1]
        assert call_kwargs["tool_filter"] is None
        conn = call_kwargs["connection_params"]
        assert conn.server_params.command == "npx"
        assert conn.server_params.args == [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/tmp",
        ]
        assert conn.server_params.env == {"KEY": "val"}

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_stdio_env_var_resolution(
        self,
        mock_toolset_cls: MagicMock,
        tmp_config: Path,
        logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MAPS_KEY", "my-api-key")
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "maps",
                        "enabled": True,
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-google-maps"],
                        "env": {"GOOGLE_MAPS_API_KEY": "${MAPS_KEY}"},
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1

        call_kwargs = mock_toolset_cls.call_args[1]
        conn = call_kwargs["connection_params"]
        assert conn.server_params.env == {"GOOGLE_MAPS_API_KEY": "my-api-key"}

    def test_stdio_missing_command_raises(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """A Stdio server without 'command' should be skipped gracefully."""
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "bad-stdio",
                        "enabled": True,
                        "type": "stdio",
                        "command": "",
                        "args": [],
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        # Should not crash – the bad server is skipped
        assert manager.get_toolsets() == []

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_stdio_with_tool_filter(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "fs-filtered",
                        "enabled": True,
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "tool_filter": ["list_directory", "read_file"],
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1

        call_kwargs = mock_toolset_cls.call_args[1]
        assert call_kwargs["tool_filter"] == ["list_directory", "read_file"]


# ---------------------------------------------------------------------------
# SSE toolset creation
# ---------------------------------------------------------------------------


class TestSseToolset:
    """Test SSE MCP server toolset creation."""

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_sse_basic(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "remote-api",
                        "enabled": True,
                        "type": "sse",
                        "url": "http://localhost:8080/sse",
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1

        call_kwargs = mock_toolset_cls.call_args[1]
        conn = call_kwargs["connection_params"]
        assert conn.url == "http://localhost:8080/sse"

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_sse_with_headers_and_env_vars(
        self,
        mock_toolset_cls: MagicMock,
        tmp_config: Path,
        logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AUTH_TOKEN", "Bearer secret")
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "authed-api",
                        "enabled": True,
                        "type": "sse",
                        "url": "http://api.example.com/sse",
                        "headers": {"Authorization": "${AUTH_TOKEN}"},
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert len(manager.get_toolsets()) == 1

        call_kwargs = mock_toolset_cls.call_args[1]
        conn = call_kwargs["connection_params"]
        assert conn.headers == {"Authorization": "Bearer secret"}

    def test_sse_missing_url_raises(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """An SSE server without 'url' should be skipped gracefully."""
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "bad-sse",
                        "enabled": True,
                        "type": "sse",
                        "url": "",
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert manager.get_toolsets() == []


# ---------------------------------------------------------------------------
# Unsupported types and error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Test graceful handling of bad configurations."""

    def test_unsupported_type_skipped(
        self, tmp_config: Path, logger: logging.Logger
    ) -> None:
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "weird",
                        "enabled": True,
                        "type": "grpc",
                        "url": "grpc://localhost",
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        assert manager.get_toolsets() == []

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_one_bad_server_does_not_block_others(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """If one server fails, the other should still load."""
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "bad",
                        "enabled": True,
                        "type": "stdio",
                        "command": "",
                        "args": [],
                    },
                    {
                        "id": "good",
                        "enabled": True,
                        "type": "stdio",
                        "command": "echo",
                        "args": ["hello"],
                    },
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        # Only the valid one should succeed
        assert len(manager.get_toolsets()) == 1

    @patch("app.services.McpService.mcp_manager.MCPToolset")
    def test_get_toolsets_returns_copy(
        self, mock_toolset_cls: MagicMock, tmp_config: Path, logger: logging.Logger
    ) -> None:
        """get_toolsets() should return a new list, not the internal one."""
        mock_toolset_cls.return_value = MagicMock()
        config_path = _write_config(
            tmp_config,
            {
                "servers": [
                    {
                        "id": "srv",
                        "enabled": True,
                        "type": "stdio",
                        "command": "echo",
                        "args": [],
                    }
                ]
            },
        )
        manager = McpManager(config_path=config_path, logger=logger)
        list_a = manager.get_toolsets()
        list_b = manager.get_toolsets()
        assert list_a is not list_b
        assert list_a == list_b

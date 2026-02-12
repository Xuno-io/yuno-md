"""
MCP (Model Context Protocol) server manager.

Loads MCP server configurations from a JSON file and creates
ADK-compatible MCPToolset instances that can be injected into the agent's
tool list. Supports both Stdio (local process) and SSE (remote HTTP)
connection types.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from google.adk.tools.mcp_tool import (
    MCPToolset,
    SseConnectionParams,
    StdioConnectionParams,
)
from mcp import StdioServerParameters


# Pattern to match ${ENV_VAR} references in config values
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    """
    Replace ``${ENV_VAR}`` placeholders in *value* with the corresponding
    environment-variable values.  Unresolved placeholders are left as-is so
    the caller can detect misconfiguration.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_env_in_dict(data: dict[str, str]) -> dict[str, str]:
    """Resolve ``${ENV_VAR}`` placeholders in every value of *data*."""
    return {k: _resolve_env_vars(v) for k, v in data.items()}


def _resolve_env_in_list(items: list[str]) -> list[str]:
    """Resolve ``${ENV_VAR}`` placeholders in every item of *items*."""
    return [_resolve_env_vars(item) for item in items]


class McpManager:
    """
    Manage MCP server configurations and create ADK toolset instances.

    The manager reads a JSON configuration file, validates each server entry,
    and builds ``MCPToolset`` instances for every **enabled** server.

    Typical usage::

        manager = McpManager(config_path="configuration/mcp_servers.json",
                             logger=my_logger)
        toolsets = manager.get_toolsets()
        # Pass *toolsets* into the ADK Agent's tools list.

    Parameters
    ----------
    config_path:
        Filesystem path to the JSON configuration file.
    logger:
        A :class:`logging.Logger` instance for diagnostics.
    """

    def __init__(self, config_path: str, logger: logging.Logger) -> None:
        self.config_path = config_path
        self.logger = logger
        self._toolsets: list[MCPToolset] = []
        self._load_and_build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_toolsets(self) -> list[MCPToolset]:
        """Return the list of successfully created ``MCPToolset`` instances."""
        return list(self._toolsets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_build(self) -> None:
        """Load the JSON config and create toolsets for enabled servers."""
        config = self._load_config()
        if config is None:
            return

        servers: list[dict[str, Any]] = config.get("servers", [])
        if not servers:
            self.logger.info("No MCP servers configured in %s", self.config_path)
            return

        for server_cfg in servers:
            server_id = server_cfg.get("id", "<unknown>")
            enabled = server_cfg.get("enabled", False)

            if not enabled:
                self.logger.debug("MCP server '%s' is disabled, skipping", server_id)
                continue

            try:
                toolset = self._build_toolset(server_cfg)
                self._toolsets.append(toolset)
                self.logger.info("MCP server '%s' loaded successfully", server_id)
            except Exception as exc:
                # Graceful degradation: one broken MCP should not crash the bot
                self.logger.error(
                    "Failed to load MCP server '%s': %s",
                    server_id,
                    exc,
                    exc_info=True,
                )

        self.logger.info(
            "McpManager ready: %d/%d MCP server(s) active",
            len(self._toolsets),
            len([s for s in servers if s.get("enabled", False)]),
        )

    def _load_config(self) -> dict[str, Any] | None:
        """Read and parse the JSON config file, returning ``None`` on error."""
        path = Path(self.config_path)
        if not path.exists():
            self.logger.warning(
                "MCP config file not found at %s – no MCP servers will be loaded",
                self.config_path,
            )
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.error(
                "Failed to read MCP config %s: %s", self.config_path, exc
            )
            return None

    def _build_toolset(self, cfg: dict[str, Any]) -> MCPToolset:
        """
        Create an ``MCPToolset`` from a single server configuration entry.

        Raises
        ------
        ValueError
            If the server type is unsupported or required fields are missing.
        """
        server_type: str = cfg.get("type", "").lower()
        tool_filter: list[str] | None = cfg.get("tool_filter")

        if server_type == "stdio":
            return self._build_stdio_toolset(cfg, tool_filter)
        if server_type == "sse":
            return self._build_sse_toolset(cfg, tool_filter)

        raise ValueError(
            f"Unsupported MCP server type '{server_type}'. "
            "Expected 'stdio' or 'sse'."
        )

    def _build_stdio_toolset(
        self,
        cfg: dict[str, Any],
        tool_filter: list[str] | None,
    ) -> MCPToolset:
        """Build an ``MCPToolset`` backed by a local Stdio process."""
        command: str = cfg.get("command", "")
        if not command:
            raise ValueError("Stdio MCP server requires a 'command' field")

        args: list[str] = _resolve_env_in_list(cfg.get("args", []))
        env: dict[str, str] | None = cfg.get("env")
        if env:
            env = _resolve_env_in_dict(env)

        connection_params = StdioConnectionParams(
            server_params=StdioServerParameters(
                command=command,
                args=args,
                env=env,
            ),
        )

        return MCPToolset(
            connection_params=connection_params,
            tool_filter=tool_filter,
        )

    def _build_sse_toolset(
        self,
        cfg: dict[str, Any],
        tool_filter: list[str] | None,
    ) -> MCPToolset:
        """Build an ``MCPToolset`` connected to a remote SSE server."""
        url: str = cfg.get("url", "")
        if not url:
            raise ValueError("SSE MCP server requires a 'url' field")

        url = _resolve_env_vars(url)
        headers: dict[str, str] | None = cfg.get("headers")
        if headers:
            headers = _resolve_env_in_dict(headers)

        connection_params = SseConnectionParams(
            url=url,
            headers=headers or {},
        )

        return MCPToolset(
            connection_params=connection_params,
            tool_filter=tool_filter,
        )

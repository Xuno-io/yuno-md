# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

yuno.md is a Telegram bot powered by Google Gemini (via Vertex AI) with long-term memory. It acts as an intellectual sparring partner — questioning premises, exposing weaknesses, and demanding rigor. The system prompt and philosophy are in Spanish.

## Environment

All commands must be run inside the `telegram-bot-adapter` conda environment:

```bash
conda activate telegram-bot-adapter
```

Or prefix any command with `conda run -n telegram-bot-adapter`.

## Commands

```bash
# Install dependencies
conda run -n telegram-bot-adapter poetry install

# Run the bot
conda run -n telegram-bot-adapter poetry run python main.py

# Initialize Telegram session (first-time setup)
conda run -n telegram-bot-adapter poetry run python init_userbot.py

# Run all tests
conda run -n telegram-bot-adapter poetry run pytest

# Run tests with coverage
conda run -n telegram-bot-adapter poetry run pytest --cov=app --cov-report=term-missing

# Run a single test file
conda run -n telegram-bot-adapter poetry run pytest tests/unit/services/test_neibot_service.py

# Run a single test
conda run -n telegram-bot-adapter poetry run pytest tests/unit/services/test_neibot_service.py::TestNeibotService::test_get_response

# Pre-commit hooks (ruff, mypy, bandit, pyupgrade, pytest)
conda run -n telegram-bot-adapter poetry run pre-commit run --all-files

# Formatting & linting (handled by ruff in pre-commit, but also available separately)
conda run -n telegram-bot-adapter poetry run black app tests
conda run -n telegram-bot-adapter poetry run isort app tests
conda run -n telegram-bot-adapter poetry run mypy app --ignore-missing-imports
conda run -n telegram-bot-adapter poetry run flake8 app tests
```

## Architecture

```
main.py → bootstrap_bot() → Components (Lagom DI) → Services
```

**Entry point**: `main.py` calls `bootstrap_bot()` which initializes DI via the `Components` singleton, then starts `TelegramService`.

### Service Layer

- **TelegramService** (`app/services/TelegramService/`) — Telethon event handlers, message routing, history management. Private chats use rolling window; group chats use thread-based replies triggered by `/yuno` or `@yunoaidotcom`.
- **NeibotService** (`app/services/NeibotService/`) — Google ADK Agent with in-memory sessions. Creates tool functions (search_memory, web_search, MCP toolsets) via closures. Handles response distillation when exceeding Telegram's 4096 char limit. Optional Langfuse tracing via `@observe`.
- **MemoryService** (`app/services/MemoryService/`) — Wraps mem0 with Redis backend. Semantic search with categories: `[TECH_STACK]`, `[BUSINESS_LOGIC]`, `[USER_CONSTRAINTS]`. Fact extraction via separate Gemini call.
- **McpManager** (`app/services/McpService/`) — Loads MCP server configs from JSON, supports Stdio and SSE connections, resolves `${ENV_VAR}` placeholders.

### Data Layer

- **Repositories** (`app/repositories/`) — `ChatRepository` for SQLite message history/model selection.
- **Database** (`app/components/database/`) — SQLite wrapper from xuno-components.
- **Entities** (`app/entities/`) — `MessagePayload`, `ImageAttachment` TypedDict.

### Dependency Injection

Uses **Lagom** DI container. `Components` class (`app/bootstrap/components.py`) is a metaclass singleton per environment. Factory functions in `app/dependencies/` create services/repositories. All services have interfaces for testability.

### Key Patterns

- **Context variables**: `contextvars.ContextVar` for thread-safe user_id tracking across async operations
- **Closures**: Tool functions (search_memory, web_search) capture service instances via closures rather than class methods
- **Rolling window**: Last 20 messages sent to LLM for context
- **Distillation**: Responses > 4096 chars are condensed via a separate model call using a 4-movement protocol

## Configuration

Environment-based JSON config in `configuration/` directory, loaded by xuno-components `Configuration`:
- `default.json` — fallback defaults
- `development.json` — dev overrides
- `mcp_servers.json` — MCP server definitions

Environment variables loaded from `.env` (see `.env-example`). Key required vars: `TELEGRAM_API_ID`, `TELEGRAM_APP_HASH`, `TELEGRAM_BOT_TOKEN`, `VERTEX_PROJECT_ID`, `GOOGLE_API_KEY`, `ADMIN_IDS`, `CREATOR_USERNAME`.

Redis must be running for memory/vector store functionality.

## Testing

- **Framework**: pytest with pytest-asyncio (auto mode — no need for `@pytest.mark.asyncio`)
- **Markers**: `unit`, `integration`, `slow`, `asyncio`
- **conftest.py** disables Langfuse tracing during tests
- Tests live in `tests/unit/` mirroring the `app/` structure

## Dependencies of Note

- `xuno-components` — Custom package (GitHub) providing `Configuration`, `Logger`, `DBInterface`
- `mem0ai` — Custom fork from `junoai-org/mem0` (feature branch for liteLLM reasoning effort)
- `google-adk` — Google Agent Development Kit for LLM orchestration with tools
- `telethon` — Telegram MTProto client (not the Bot API)

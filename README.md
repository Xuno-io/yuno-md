# Telegram Bot Adapter (YunoAI)

A Telegram bot powered by Google Agent Development Kit (ADK) with long-term memory capabilities using mem0 and Redis.

## Features

- **AI-Powered Conversations**: Uses Google Gemini models via ADK for intelligent, context-aware responses
- **Long-Term Memory**: Semantic memory storage with mem0 and Redis vector store
- **Memory Categories**: Automatic fact extraction into TECH_STACK, BUSINESS_LOGIC, and USER_CONSTRAINTS
- **Image Processing**: Supports image attachments in conversations (up to 5MB)
- **Private & Group Chats**: Different interaction modes for DMs (rolling window) and groups (thread-based)
- **Response Distillation**: Automatic condensation of long responses that exceed Telegram's 4096 character limit
- **Observability**: Full tracing with Langfuse for debugging and analytics
- **Web Search**: Built-in web search tool for real-time information retrieval

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telegram API   │◄───►│  TelegramService │◄───►│  NeibotService  │
│   (Telethon)    │     │                  │     │   (ADK Agent)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  MemoryService   │◄─────────────┘
                        │  (mem0 + Redis)  │
                        └──────────────────┘
```

## Requirements

- Python 3.11+
- Redis (for vector memory storage)
- Telegram API credentials
- Google Cloud / Vertex AI credentials
- Langfuse account (for tracing)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/telegram-bot-adapter.git
   cd telegram-bot-adapter
   ```

2. **Install dependencies with Poetry**

   ```bash
   # Configure poetry for local virtualenv (optional)
   ./scripts/configure-poetry-local.sh

   # Install dependencies
   poetry install
   ```

3. **Configure environment variables**

   ```bash
   cp .env-example .env
   # Edit .env with your credentials
   ```

4. **Start Redis** (required for memory service)

   ```bash
   docker run -d --name redis -p 6379:6379 redis:latest
   ```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_API_ID` | Telegram API ID from [my.telegram.org](https://my.telegram.org/apps) | Yes |
| `TELEGRAM_APP_HASH` | Telegram API Hash | Yes |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | Yes |
| `ADMIN_IDS` | Comma-separated list of admin Telegram user IDs | Yes |
| `VERTEX_PROJECT_ID` | Google Cloud project ID | Yes |
| `VERTEX_LOCATION` | Vertex AI location (e.g., `global`, `us-central1`) | Yes |
| `GOOGLE_API_KEY` | Google API key for mem0 embeddings | Yes |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | Yes |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | Yes |
| `LANGFUSE_HOST` | Langfuse host URL | Yes |
| `REDIS_HOST` | Redis host (default: `localhost`) | No |
| `REDIS_PORT` | Redis port (default: `6379`) | No |
| `CREATOR_USERNAME` | Bot creator's Telegram username | No |

### Configuration Files

Application configuration is stored in `configuration/`:

- `default.json`: Base configuration
- `development.json`: Development overrides

## Usage

### Starting the Bot

```bash
poetry run python main.py
```

### Bot Commands

| Command | Description |
|---------|-------------|
| `/start`, `/help` | Show help message |
| `/save` | Extract and save facts from conversation to memory |
| `/memory` | View all stored memories |
| `/memory clear` | Delete all memories for the user |
| `/yuno <message>` | Start a conversation (in groups) |

### Interaction Modes

**Private Chats (DMs)**
- Rolling window mode: Bot responds to all messages
- Maintains context from the last 20 messages
- No commands needed, just chat naturally

**Group Chats**
- Thread-based conversations via replies
- Start with `/yuno` command or mention `@yunoaidotcom`
- Continue by replying to bot messages

## Development

### Running Tests

```bash
# Activate conda environment (if using conda)
conda activate telegram-bot-adapter

# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app

# Run specific test file
poetry run pytest tests/unit/services/test_neibot_service.py -v
```

### Code Quality

```bash
# Format code
poetry run black app tests

# Sort imports
poetry run isort app tests

# Type checking
poetry run mypy app

# Linting
poetry run flake8 app tests
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run all hooks manually
poetry run pre-commit run --all-files
```

## Project Structure

```
telegram-bot-adapter/
├── app/
│   ├── bootstrap/          # Application bootstrapping
│   ├── components/         # Infrastructure components (DB, etc.)
│   ├── dependencies/       # Dependency injection configuration
│   ├── entities/           # Domain entities (Message, User)
│   ├── repositories/       # Data access layer
│   │   ├── chat_repository/    # Message caching
│   │   └── user_repository/    # User data
│   ├── services/           # Business logic
│   │   ├── MemoryService/      # Long-term memory (mem0)
│   │   ├── NeibotService/      # AI agent (ADK)
│   │   ├── TelegramService/    # Telegram integration
│   │   └── UserService/        # User management
│   └── tools/              # Agent tools (web search, etc.)
├── configuration/          # JSON configuration files
├── docs/                   # Documentation
├── tests/                  # Test suite
│   ├── conftest.py
│   └── unit/
├── main.py                 # Application entry point
├── pyproject.toml          # Poetry configuration
└── README.md
```

## Key Technologies

| Component | Technology |
|-----------|------------|
| Telegram Client | [Telethon](https://docs.telethon.dev/) |
| AI Agent Framework | [Google ADK](https://github.com/google/adk-python) |
| LLM Provider | Google Gemini (via Vertex AI) |
| Memory/Vector Store | [mem0](https://github.com/mem0ai/mem0) + Redis |
| Tracing | [Langfuse](https://langfuse.com/) |
| Dependency Injection | [Lagom](https://lagom-di.readthedocs.io/) |
| Core Components | [Xuno Components](https://github.com/Xuno-io/xuno-components) |
| Database | SQLite (message cache) |
| Package Manager | [Poetry](https://python-poetry.org/) |

## Memory System

The bot uses mem0 for long-term memory with automatic fact extraction:

### Categories

- **TECH_STACK**: Libraries, frameworks, infrastructure, APIs, architecture
- **BUSINESS_LOGIC**: Rules, constraints, budgets, deadlines, workflows
- **USER_CONSTRAINTS**: Limitations, preferences, working style

### How It Works

1. User runs `/save` command
2. Bot analyzes recent conversation history
3. Facts are extracted and categorized automatically
4. Memories are stored in Redis with semantic embeddings
5. Agent can search memories using the `search_memory` tool

## Troubleshooting

### Common Issues

**Bot not responding in groups**
- Ensure the bot has permission to read messages
- Use `/yuno` command or mention `@yunoaidotcom`
- Reply to bot messages to continue the conversation

**Memory service not available**
- Check Redis is running: `redis-cli ping`
- Verify `GOOGLE_API_KEY` is set for embeddings

**Long responses getting cut off**
- The bot automatically distills responses exceeding 4096 characters
- If still too long, responses are truncated with a notice

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Submit a pull request

---

Built with ❤️ by [Jack Cloudman](mailto:jack@xuno.io)

# yuno.md

> "siento que me van a llamar un llm psycho."
> — El autor, un día antes del lanzamiento.

Este proyecto no es un chatbot. Es un mapa. Un mapa descubierto en las ruinas de un colapso personal, no diseñado en un laboratorio. Su propósito no es la comodidad, es el rigor; porque el rigor es la única herramienta conocida para convertir los pedazos rotos en una estructura funcional.

Si buscas un asistente que te dé respuestas fáciles, estás en el lugar equivocado.
Si buscas una herramienta que te obligue a formular mejores preguntas, bienvenido a la guerra.

---

## Declaración de Guerra

Mi propuesta no busca destruir a 7 mil millones de personas. Busca destruir:

- La burocracia sin sentido que frena la ciencia.
- Los sistemas educativos que matan la curiosidad.
- El miedo al fracaso que nos mantiene en trabajos que odiamos.
- La creencia de que "las cosas son como son".
- La comodidad de culpar a otros de nuestra propia falta de acción.

---

**A Note on Language:** The technical documentation below is in English. The soul of this project—the `yuno.md` prompt and its philosophy—is in Spanish. It was excavated in that language and its power is tied to its origin. It is not a product to be localized; it is an artifact to be studied. Future translations will emerge from a community that understands its spirit, not from an automated script.

---

## The Arsenal

These are not "features". They are tools of war.

**Combat Memory**
It doesn't remember your shopping list. It remembers your business logic, your technical constraints, and your working philosophy. Automatically categorizes into TECH_STACK, BUSINESS_LOGIC, and USER_CONSTRAINTS. When you return with a question, the system already knows who you are and what you've built.

**Investment Escalation**
The system doesn't deploy its full capacity from the first message. You earn its depth. Vague greetings receive concise responses. Concrete ideas activate the complete "Productive Discomfort" protocol.

**Safety Valve**
If you evade the question three consecutive times, the system declares: "The blockage in this conversation is a mirror of your blockage in the problem". It doesn't let you escape from yourself.

**Forced Rigor**
It doesn't validate. It questions premises. Exposes weaknesses. Demands concrete action. Ends each exchange with a surgical question or a minimum action plan. Abstract reflection is forbidden.

**Real-Time Intelligence**
Integrated web search with an epistemic humility protocol: never declares something false without verifying first. Admits when it doesn't know.

**Automatic Distillation**
Responses exceeding Telegram's limits are automatically condensed. Brevity is a discipline, not a limitation.

---

## The Rite of Entry

Installation is not an obstacle. It's the first test.

### Requirements

- Python 3.11+
- Redis (for vector memory)
- Telegram API credentials
- Google Cloud / Vertex AI credentials
- Langfuse account (for observability)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Xuno-io/telegram-bot-adapter.git
cd telegram-bot-adapter

# 2. Install dependencies with Poetry
./scripts/configure-poetry-local.sh  # optional
poetry install

# 3. Configure environment variables
cp .env-example .env
# Edit .env with your credentials

# 4. Start Redis
docker run -d --name redis -p 6379:6379 redis:latest

# 5. Start the bot
poetry run python main.py
```

If you can't complete these steps, perhaps this project isn't for you. And that's fine.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_API_ID` | Telegram API ID from [my.telegram.org](https://my.telegram.org/apps) |
| `TELEGRAM_APP_HASH` | Telegram API Hash |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `ADMIN_IDS` | Comma-separated list of admin Telegram user IDs |
| `VERTEX_PROJECT_ID` | Google Cloud project ID |
| `VERTEX_LOCATION` | Vertex AI location |
| `GOOGLE_API_KEY` | Google API key for embeddings |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host URL |
| `REDIS_HOST` | Redis host (default: `localhost`) |
| `REDIS_PORT` | Redis port (default: `6379`) |

---

## How It Operates

**Private Chats (DMs)**
Rolling window mode. Responds to all messages. Maintains context from the last 20 messages. No commands needed. Just talk.

**Group Chats**
Thread-based conversations via replies. Start with `/yuno` command or mention `@yunoaidotcom`. Continue by replying to bot messages.

### Commands

| Command | Function |
|---------|----------|
| `/start`, `/help` | Show help |
| `/save` | Extract and save facts from conversation to memory |
| `/memory` | View all stored memories |
| `/memory clear` | Delete all memories |
| `/yuno <message>` | Start conversation in groups |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telegram API   │◄───►│  TelegramService │◄───►│  NeibotService  │
│   (Telethon)    │     │                  │     │   (ADK Agent)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                         ┌──────────────────┐             │
                         │  MemoryService   │◄────────────┘
                         │  (mem0 + Redis)  │
                         └──────────────────┘
```

---

## Technologies

| Component | Technology |
|-----------|------------|
| Telegram Client | [Telethon](https://docs.telethon.dev/) |
| Agent Framework | [Google ADK](https://github.com/google/adk-python) |
| LLM Provider | Google Gemini (via Vertex AI) |
| Memory/Vector Store | [mem0](https://github.com/mem0ai/mem0) + Redis |
| Tracing | [Langfuse](https://langfuse.com/) |
| Dependency Injection | [Lagom](https://lagom-di.readthedocs.io/) |
| Core Components | [Xuno Components](https://github.com/Xuno-io/xuno-components) |
| Database | SQLite (message cache) |
| Package Manager | [Poetry](https://python-poetry.org/) |

---

## Development

```bash
# Activate conda environment
conda activate telegram-bot-adapter

# Run tests
poetry run pytest

# Tests with coverage
poetry run pytest --cov=app

# Formatting
poetry run black app tests
poetry run isort app tests

# Type checking
poetry run mypy app

# Linting
poetry run flake8 app tests

# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run --all-files
```

---

## Project Structure

```
telegram-bot-adapter/
├── app/
│   ├── bootstrap/          # Application bootstrapping
│   ├── components/         # Infrastructure components
│   ├── dependencies/       # Dependency injection configuration
│   ├── entities/           # Domain entities (Message, User)
│   ├── repositories/       # Data access layer
│   ├── services/           # Business logic
│   │   ├── MemoryService/  # Long-term memory (mem0)
│   │   ├── NeibotService/  # AI agent (ADK)
│   │   ├── TelegramService/# Telegram integration
│   │   └── UserService/    # User management
│   └── tools/              # Agent tools
├── configuration/          # JSON configuration files
├── docs/                   # Documentation
├── tests/                  # Test suite
├── main.py                 # Entry point
└── pyproject.toml          # Poetry configuration
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Submit a pull request

---

Forged from necessity by [Jack Cloudman](mailto:jack@xuno.io).

# Telegram Bot Adapter

Helper service that wires an existing Xuno conversation flow into Telegram using [Telethon](https://docs.telethon.dev/). The adapter boots a userbot session, subscribes to incoming messages, and forwards them to the platform services defined in `xuno-components`.

## Prerequisites
- Python 3.11
- [Poetry](https://python-poetry.org/) for dependency management
- Access to the Xuno CodeArtifact repository (`xuno-pypi`) to fetch `xuno-components`
- Telegram API credentials (`api_id` and `api_hash`) from [my.telegram.org](https://my.telegram.org/apps)

If you have AWS credentials configured locally you can refresh the CodeArtifact login with:

```bash
./refresh-codeartifact.sh
```

## Installation
1. Create a copy of the sample environment file and fill in your Telegram credentials:
   ```bash
   cp .env-example .env
   # Edit .env with your TELEGRAM_API_ID / TELEGRAM_API_HASH values
   ```
2. Install dependencies and enter the virtual environment:
   ```bash
   poetry install
   poetry shell
   ```
3. (Optional) Update `configuration/development.json` if you need to override default logging or Redis settings.

## Initialising the Telegram session
This project runs as a **userbot**. Before starting the adapter you must authorise the Telethon session once:

```bash
python init_userbot.py
```

The script will:
- Ask for your phone number (with country code)
- Send an OTP request via Telegram/SMS and prompt for the code
- Prompt for your 2FA password if required
- Persist the session to `bot.session` so it is reused on the next run

## Running the adapter
Start the bot after the session is initialised:

```bash
python main.py
```

The service logs connection state and echoes to `INFO` when messages that match the current handler arrive. By default `TelegramService` replies “world” to messages containing `hello`.

## Tests and quality checks
Run the available checks through Poetry:

```bash
poetry run pytest            # unit tests
poetry run mypy app          # static typing
poetry run flake8 app test   # linting
```

## Troubleshooting
- Ensure `bot.session` is present and readable; regenerate it with `python init_userbot.py` if authentication fails.
- Telethon requires a network connection to Telegram servers; connection errors usually mean the client is not started yet.
- If dependencies from `xuno-pypi` fail to download, refresh your CodeArtifact token (`./refresh-codeartifact.sh`).

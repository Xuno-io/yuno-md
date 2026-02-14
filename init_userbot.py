import asyncio

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

from app.dependencies.components import get_components


async def ensure_authorized(client: TelegramClient) -> None:
    if await client.is_user_authorized():
        return

    phone: str = input(
        "Enter your phone number with country code (e.g. +34123456789): "
    )
    await client.send_code_request(phone)

    code: str = input("Enter the code you received via SMS or Telegram: ")
    password: str | None = None

    try:
        await client.sign_in(phone=phone, code=code)
    except SessionPasswordNeededError:
        password = input("Your account has 2FA enabled, enter your password: ")
        await client.sign_in(password=password)


async def authorize_bot(client: TelegramClient) -> None:
    token: str = input("Enter the bot token (from @BotFather): ")
    await client.sign_in(bot_token=token)


async def main() -> None:
    print("What type of session do you want to initialize?")
    print("1. User bot (personal account)")
    print("2. Traditional bot (@BotFather token)")
    choice: str = input("Select an option (1/2): ").strip()

    if choice not in ("1", "2"):
        print("Invalid option.")
        return

    components = get_components(env="development", config_path="configuration")
    client: TelegramClient = components.get_component(TelegramClient)

    # Telethon handles the session and creates the .session file automatically.
    await client.connect()

    if choice == "1":
        await ensure_authorized(client)
    else:
        await authorize_bot(client)

    me = await client.get_me()
    name = me.username or me.first_name or "<unnamed>"
    print(f"Session started as: {name}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

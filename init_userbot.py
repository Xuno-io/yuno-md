import asyncio

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

from app.dependencies.components import get_components


async def ensure_authorized(client: TelegramClient) -> None:
    if await client.is_user_authorized():
        return

    phone: str = input(
        "Ingresa tu número de teléfono con código de país (e.g. +34123456789): "
    )
    await client.send_code_request(phone)

    code: str = input("Ingresa el código que recibiste por SMS o Telegram: ")
    password: str | None = None

    try:
        await client.sign_in(phone=phone, code=code)
    except SessionPasswordNeededError:
        password = input("Tu cuenta tiene 2FA, ingresa tu contraseña: ")
        await client.sign_in(password=password)


async def main() -> None:
    components = get_components(env="development", config_path="configuration")
    client: TelegramClient = components.get_component(TelegramClient)

    # Telethon maneja la sesión y crea el archivo .session automáticamente.
    await client.connect()
    await ensure_authorized(client)

    me = await client.get_me()
    nombre = me.username or me.first_name or "<sin nombre>"
    print(f"Sesión iniciada como: {nombre}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

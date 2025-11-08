from app.dependencies.components import get_components
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.TelegramService.telegram_service_interface import (
    TelegramServiceInterface,
)
from app.dependencies.services import get_neibot_dspy_service, get_telegram_service


async def bootstrap_bot(
    env: str = "development",
    config_path: str = "configuration",
) -> TelegramServiceInterface:
    components = get_components(env=env, config_path=config_path)
    neibot: NeibotServiceInterface = get_neibot_dspy_service(components)

    bot: TelegramServiceInterface = await get_telegram_service(components, neibot)
    return bot

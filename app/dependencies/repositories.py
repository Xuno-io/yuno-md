from app.bootstrap.components import Components
from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)
from app.repositories.chat_repository.sqlite_chat_repository import SqliteChatRepository
from xuno_components.database.db_interface import DBInterface


def get_chat_repository(components: Components) -> ChatRepositoryInterface:
    return SqliteChatRepository(components.get_component(DBInterface))

from app.bootstrap.components import Components
from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)
from app.repositories.chat_repository.sqlite_chat_repository import SqliteChatRepository
from app.repositories.user_repository.user_repository_interface import (
    UserRepositoryInterface,
)
from app.repositories.user_repository.sqlite_user_repository import SqliteUserRepository
from xuno_components.database.db_interface import DBInterface


def get_chat_repository(components: Components) -> ChatRepositoryInterface:
    return SqliteChatRepository(components.get_component(DBInterface))


def get_user_repository(components: Components) -> UserRepositoryInterface:
    return SqliteUserRepository(components.get_component(DBInterface))

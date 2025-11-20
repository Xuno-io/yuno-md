from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)
from xuno_components.database.db_interface import DBInterface


class SqliteChatRepository(ChatRepositoryInterface):
    def __init__(self, db: DBInterface):
        self.db = db
        self._init_table()

    def _init_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS chat_configurations (
            chat_id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL
        );
        """
        self.db.execute(query)

    def get_configuration(self) -> dict:
        # Not used for now, but required by interface
        return {}

    def get_model(self, chat_id: int) -> str | None:
        query = "SELECT model_name FROM chat_configurations WHERE chat_id = ?"
        result = self.db.execute_and_fetchone(query, (chat_id,))
        if result:
            return result["model_name"]
        return None

    def set_model(self, chat_id: int, model: str) -> None:
        query = """
        INSERT INTO chat_configurations (chat_id, model_name)
        VALUES (?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET model_name = excluded.model_name;
        """
        self.db.execute(query, (chat_id, model))

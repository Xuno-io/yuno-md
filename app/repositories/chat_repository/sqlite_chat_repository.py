import json
from app.repositories.chat_repository.chat_repository_interface import (
    ChatRepositoryInterface,
)
from app.entities.message import MessagePayload
from xuno_components.database.db_interface import DBInterface


class SqliteChatRepository(ChatRepositoryInterface):
    def __init__(self, db: DBInterface):
        self.db = db
        self._init_table()

    def _init_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS chat_configurations (
            chat_id BIGINT PRIMARY KEY,
            model_name TEXT NOT NULL
        );
        """
        self.db.execute(query)

        query_messages = """
        CREATE TABLE IF NOT EXISTS messages (
            chat_id BIGINT,
            message_id BIGINT,
            reply_to_msg_id BIGINT,
            role TEXT,
            content TEXT,
            attachments TEXT,
            PRIMARY KEY (chat_id, message_id)
        );
        """
        self.db.execute(query_messages)

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

    def get_message(self, chat_id: int, message_id: int) -> dict | None:
        query = """
        SELECT reply_to_msg_id, role, content, attachments
        FROM messages
        WHERE chat_id = ? AND message_id = ?
        """
        result = self.db.execute_and_fetchone(query, (chat_id, message_id))
        if result:
            return {
                "payload": {
                    "role": result["role"],
                    "content": result["content"],
                    "attachments": json.loads(result["attachments"]),
                },
                "reply_to_msg_id": result["reply_to_msg_id"],
            }
        return None

    def save_message(
        self,
        chat_id: int,
        message_id: int,
        payload: MessagePayload,
        reply_to_msg_id: int | None,
    ) -> None:
        query = """
        INSERT INTO messages (chat_id, message_id, reply_to_msg_id, role, content, attachments)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, message_id) DO UPDATE SET
            reply_to_msg_id = excluded.reply_to_msg_id,
            role = excluded.role,
            content = excluded.content,
            attachments = excluded.attachments;
        """
        self.db.execute(
            query,
            (
                chat_id,
                message_id,
                reply_to_msg_id,
                payload["role"],
                payload["content"],
                json.dumps(payload["attachments"]),
            ),
        )

from app.repositories.user_repository.user_repository_interface import (
    UserRepositoryInterface,
)
from app.entities.user import User
from xuno_components.database.db_interface import DBInterface


class SqliteUserRepository(UserRepositoryInterface):
    def __init__(self, db: DBInterface):
        self.db = db
        self._init_table()

    def _init_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            is_pro BOOLEAN NOT NULL DEFAULT 0
        );
        """
        self.db.execute(query)

    def get_user(self, user_id: int) -> User | None:
        query = "SELECT user_id, is_pro FROM users WHERE user_id = ?"
        result = self.db.execute_and_fetchone(query, (user_id,))
        if result:
            return {
                "user_id": result["user_id"],
                "is_pro": bool(result["is_pro"]),
            }
        return None

    def set_user_pro_status(self, user_id: int, is_pro: bool) -> None:
        query = """
        INSERT INTO users (user_id, is_pro)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET is_pro = excluded.is_pro;
        """
        self.db.execute(query, (user_id, is_pro))

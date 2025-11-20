import sqlite3
from typing import Any
from xuno_components.database.db_interface import DBInterface


class SqliteDB(DBInterface):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

    def connect(self) -> None:
        """
        Establish a connection to the database.
        """
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()

    def close(self) -> None:
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None

    def execute(self, query: str, params: tuple | None = None) -> None:
        """
        Execute a query with no return value.
        """
        if self.cursor is None:
            raise ConnectionError("Database not connected")

        if params is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, params)

    def execute_and_fetch(
        self, query: str, params: tuple | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a query and fetch all results as a list of dictionaries.
        """
        if self.cursor is None:
            raise ConnectionError("Database not connected")

        if params is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, params)

        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def execute_and_fetchone(
        self, query: str, params: tuple | None = None
    ) -> dict[str, Any] | None:
        """
        Execute a query and fetch a single result as a dictionary.
        """
        if self.cursor is None:
            raise ConnectionError("Database not connected")

        if params is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, params)

        row = self.cursor.fetchone()
        if row:
            return dict(row)
        return None

    def begin_transaction(self) -> None:
        """
        Begin a database transaction.
        """
        if self.cursor is None:
            raise ConnectionError("Database not connected")
        self.cursor.execute("BEGIN TRANSACTION")

    def commit(self) -> None:
        """
        Commit the current transaction.
        """
        if self.connection is None:
            raise ConnectionError("Database not connected")
        self.connection.commit()

    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """
        if self.connection is None:
            raise ConnectionError("Database not connected")
        self.connection.rollback()

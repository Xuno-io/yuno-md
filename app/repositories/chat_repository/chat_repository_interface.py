from app.entities.message import MessagePayload


class ChatRepositoryInterface:
    def get_configuration(self) -> dict:
        """Retrieve chat configuration settings."""
        raise NotImplementedError

    def get_model(self, chat_id: int) -> str | None:
        """Retrieve the model configured for a specific chat."""
        raise NotImplementedError

    def set_model(self, chat_id: int, model: str) -> None:
        """Set the model for a specific chat."""
        raise NotImplementedError

    def get_message(self, chat_id: int, message_id: int) -> dict | None:
        """Retrieve a message from the history.

        Returns a mapping with keys `payload` (MessagePayload) and
        `reply_to_msg_id` (optional int).
        """
        raise NotImplementedError

    def save_message(
        self,
        chat_id: int,
        message_id: int,
        payload: MessagePayload,
        reply_to_msg_id: int | None,
    ) -> None:
        """Save a message to the history."""
        raise NotImplementedError

    def get_recent_messages(self, chat_id: int, limit: int) -> list[dict]:
        """Retrieve the most recent messages from a chat.

        Returns a list of dicts with keys `message_id`, `payload` (MessagePayload),
        and `reply_to_msg_id`. Ordered by message_id descending (newest first).
        """
        raise NotImplementedError

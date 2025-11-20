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

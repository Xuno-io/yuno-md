from abc import ABC, abstractmethod


class UserServiceInterface(ABC):
    @abstractmethod
    def get_user_max_history_turns(self, user_id: int) -> int:
        pass

    @abstractmethod
    def set_user_pro_status(self, user_id: int, is_pro: bool) -> None:
        pass

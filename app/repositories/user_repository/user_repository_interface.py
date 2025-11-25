from abc import ABC, abstractmethod
from app.entities.user import User


class UserRepositoryInterface(ABC):
    @abstractmethod
    def get_user(self, user_id: int) -> User | None:
        pass

    @abstractmethod
    def set_user_pro_status(self, user_id: int, is_pro: bool) -> None:
        pass

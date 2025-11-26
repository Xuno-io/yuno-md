from app.services.UserService.user_service_interface import UserServiceInterface
from app.repositories.user_repository.user_repository_interface import (
    UserRepositoryInterface,
)


class UserService(UserServiceInterface):
    def __init__(
        self,
        user_repository: UserRepositoryInterface,
        default_max_history_turns: int,
        pro_max_history_turns: int,
    ):
        self.user_repository = user_repository
        self.default_max_history_turns = default_max_history_turns
        self.pro_max_history_turns = pro_max_history_turns

    def get_user_max_history_turns(self, user_id: int) -> int:
        if self.is_user_pro(user_id):
            return self.pro_max_history_turns
        return self.default_max_history_turns

    def is_user_pro(self, user_id: int) -> bool:
        user = self.user_repository.get_user(user_id)
        return bool(user and user["is_pro"])

    def set_user_pro_status(self, user_id: int, is_pro: bool) -> None:
        self.user_repository.set_user_pro_status(user_id, is_pro)

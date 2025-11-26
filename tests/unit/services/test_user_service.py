import unittest
from unittest.mock import MagicMock
from app.services.UserService.user_service import UserService
from app.repositories.user_repository.user_repository_interface import (
    UserRepositoryInterface,
)


class TestUserService(unittest.TestCase):
    def setUp(self):
        self.mock_repo = MagicMock(spec=UserRepositoryInterface)
        self.default_limit = 100
        self.pro_limit = 200
        self.service = UserService(self.mock_repo, self.default_limit, self.pro_limit)

    def test_get_user_max_history_turns_default(self):
        self.mock_repo.get_user.return_value = None
        limit = self.service.get_user_max_history_turns(123)
        self.assertEqual(limit, self.default_limit)

    def test_get_user_max_history_turns_not_pro(self):
        self.mock_repo.get_user.return_value = {"user_id": 123, "is_pro": False}
        limit = self.service.get_user_max_history_turns(123)
        self.assertEqual(limit, self.default_limit)

    def test_get_user_max_history_turns_pro(self):
        self.mock_repo.get_user.return_value = {"user_id": 123, "is_pro": True}
        limit = self.service.get_user_max_history_turns(123)
        self.assertEqual(limit, self.pro_limit)

    def test_set_user_pro_status(self):
        self.service.set_user_pro_status(123, True)
        self.mock_repo.set_user_pro_status.assert_called_once_with(123, True)

    def test_is_user_pro_true(self):
        self.mock_repo.get_user.return_value = {"user_id": 123, "is_pro": True}
        self.assertTrue(self.service.is_user_pro(123))

    def test_is_user_pro_false(self):
        self.mock_repo.get_user.return_value = {"user_id": 123, "is_pro": False}
        self.assertFalse(self.service.is_user_pro(123))

    def test_is_user_pro_none(self):
        self.mock_repo.get_user.return_value = None
        self.assertFalse(self.service.is_user_pro(123))


if __name__ == "__main__":
    unittest.main()

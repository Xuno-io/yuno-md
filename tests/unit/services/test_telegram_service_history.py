import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from app.services.TelegramService.telegram_service import TelegramService


@pytest.fixture
def mock_chat_repo():
    repo = MagicMock()
    repo.get_message.return_value = None
    repo.save_message.return_value = None
    repo.get_model.return_value = None
    return repo


@pytest.fixture
def mock_client():
    client = AsyncMock()
    return client


@pytest.fixture
def telegram_service(mock_chat_repo, mock_client):
    user_service = MagicMock()
    user_service.get_user_max_history_turns.return_value = 100
    user_service.is_user_pro.return_value = False
    service = TelegramService(
        command_prefix="/cmd",
        neibot=AsyncMock(),
        telegram_client=mock_client,
        logger=MagicMock(),
        chat_repository=mock_chat_repo,
        user_service=user_service,
        admin_ids=[123],
    )
    service.me = MagicMock()
    service.me.id = 999
    return service


@pytest.mark.asyncio
async def test_build_reply_history_fetches_and_saves(
    telegram_service, mock_chat_repo, mock_client
):
    # Setup
    event = MagicMock()
    event.reply_to_msg_id = 100
    event.chat_id = 123
    event.sender_id = 789

    # Mock message 100
    msg100 = AsyncMock()
    msg100.sender_id = 456
    msg100.raw_text = "Hello"
    msg100.reply_to_msg_id = None
    msg100.get_sender.return_value = MagicMock(
        first_name="Alice", last_name="", username="alice"
    )
    msg100.get_chat.return_value = MagicMock(title="Chat")
    msg100.download_media = AsyncMock(return_value=None)  # No media
    msg100.media = None
    msg100.photo = None
    msg100.date = datetime(2025, 11, 29, 14, 30)

    mock_client.get_messages.return_value = msg100

    # Execute
    max_history_turns = 100
    history = await telegram_service._TelegramService__build_reply_history(
        event, max_history_turns
    )

    # Verify
    assert len(history) == 1
    assert history[0]["content"] == "[Group: Chat][User: alice]: Hello"

    # Verify repository calls
    mock_chat_repo.get_message.assert_called_once_with(123, 100)
    mock_chat_repo.save_message.assert_called_once()
    args = mock_chat_repo.save_message.call_args
    assert args[0][0] == 123  # chat_id
    assert args[0][1] == 100  # message_id
    assert args[0][2]["content"] == "[Group: Chat][User: alice]: Hello"
    assert args[0][3] is None  # reply_to_msg_id


@pytest.mark.asyncio
async def test_build_reply_history_uses_cache(
    telegram_service, mock_chat_repo, mock_client
):
    # Setup
    event = MagicMock()
    event.reply_to_msg_id = 200
    event.chat_id = 123
    event.sender_id = 789

    # Mock cache hit for 200
    cached_payload = {"role": "user", "content": "Cached Message", "attachments": []}
    mock_chat_repo.get_message.side_effect = [
        {
            "payload": cached_payload,
            "reply_to_msg_id": 100,
        },  # First call returns cached
        None,  # Second call (for 100) returns None, so fetch
    ]

    # Mock message 100 (fetched)
    msg100 = AsyncMock()
    msg100.sender_id = 456
    msg100.raw_text = "Fetched Message"
    msg100.reply_to_msg_id = None
    msg100.get_sender.return_value = MagicMock(
        first_name="Bob", last_name="", username="bob"
    )
    msg100.get_chat.return_value = MagicMock(title="Chat")
    msg100.media = None
    msg100.photo = None
    msg100.date = datetime(2025, 11, 29, 14, 30)

    mock_client.get_messages.return_value = msg100

    # Execute
    max_history_turns = 100
    history = await telegram_service._TelegramService__build_reply_history(
        event, max_history_turns
    )

    # Verify
    assert len(history) == 2
    assert (
        history[0]["content"] == "[Group: Chat][User: bob]: Fetched Message"
    )  # Inserted at 0
    assert history[1]["content"] == "Cached Message"

    # Verify repository calls
    assert mock_chat_repo.get_message.call_count == 2
    mock_chat_repo.get_message.assert_any_call(123, 200)
    mock_chat_repo.get_message.assert_any_call(123, 100)

    # Verify save called for fetched message
    mock_chat_repo.save_message.assert_called_once()
    args = mock_chat_repo.save_message.call_args
    assert args[0][1] == 100

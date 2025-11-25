import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.TelegramService.telegram_service import TelegramService


@pytest.fixture
def mock_chat_repo():
    return MagicMock()


@pytest.fixture
def mock_client():
    return AsyncMock()


@pytest.fixture
def telegram_service(mock_chat_repo, mock_client):
    user_service = MagicMock()
    user_service.get_user_max_history_turns.return_value = 100
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
async def test_cached_message_does_not_trigger_download(
    telegram_service, mock_chat_repo, mock_client
):
    # Setup
    event = MagicMock()
    event.reply_to_msg_id = 200
    event.chat_id = 123
    event.sender_id = 789

    # Mock cache hit for 200 with attachments
    cached_payload = {
        "role": "user",
        "content": "Cached Message",
        "attachments": [
            {
                "base64": "stored_base64",
                "mime_type": "image/png",
                "size_bytes": 100,
                "file_name": "image.png",
            }
        ],
    }

    # Mock cache miss for 100 (parent of 200)
    mock_chat_repo.get_message.side_effect = [
        {"payload": cached_payload, "reply_to_msg_id": 100},
        None,
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
    msg100.download_media = AsyncMock(return_value=None)  # Should be called
    msg100.media = None
    msg100.photo = None

    mock_client.get_messages.return_value = msg100

    # Execute
    max_history_turns = 100
    history = await telegram_service._TelegramService__build_reply_history(event, max_history_turns)

    # Verify
    assert len(history) == 2

    # Message 200 (Cached)
    assert history[1]["content"] == "Cached Message"
    assert history[1]["attachments"][0]["base64"] == "stored_base64"

    # Verify get_messages was called ONLY for msg100 (ids=100), not for msg200
    mock_client.get_messages.assert_called_once_with(123, ids=100)

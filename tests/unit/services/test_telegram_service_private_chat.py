import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.TelegramService.telegram_service import TelegramService


@pytest.fixture
def mock_neibot():
    neibot = AsyncMock()
    neibot.get_response.return_value = "Response"
    return neibot


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get_me.return_value = AsyncMock(id=12345)
    return client


@pytest.fixture
async def telegram_service(mock_neibot, mock_client):
    # Mock chat_repository with required methods
    mock_chat_repo = MagicMock()
    mock_chat_repo.get_recent_messages.return_value = []  # Cache miss -> fallback to Telegram
    mock_chat_repo.save_message = MagicMock()  # No-op for saving

    service = TelegramService(
        command_prefix="/yo",
        neibot=mock_neibot,
        telegram_client=mock_client,
        logger=MagicMock(),
        chat_repository=mock_chat_repo,
        user_service=MagicMock(),
        admin_ids=[999],
    )
    # Don't call start() - it blocks on run_until_disconnected()
    # Instead, set self.me directly from mock_client
    service.me = await mock_client.get_me()
    return service


@pytest.mark.asyncio
async def test_is_private_chat_returns_true_for_dm(telegram_service):
    """Verify detection of private chats using Telethon's is_private property."""
    event = AsyncMock()
    event.is_private = True

    is_private = await telegram_service._is_private_chat(event)
    assert is_private is True


@pytest.mark.asyncio
async def test_is_private_chat_returns_false_for_group(telegram_service):
    """Verify detection of group chats using Telethon's is_private property."""
    event = AsyncMock()
    event.is_private = False

    is_private = await telegram_service._is_private_chat(event)
    assert is_private is False


@pytest.mark.asyncio
async def test_is_private_chat_returns_false_on_exception(telegram_service):
    """Verify fallback to False when is_private raises an exception."""

    # Use a local class to avoid polluting AsyncMock's class definition
    class FailingEvent:
        @property
        def is_private(self):
            raise Exception("test error")

    event = FailingEvent()

    is_private = await telegram_service._is_private_chat(event)
    assert is_private is False


@pytest.mark.asyncio
async def test_should_respond_true_in_private_chat_without_command(telegram_service):
    """Verify implicit response in private chat."""
    event = AsyncMock()
    event.raw_text = "hola yuno"
    event.is_private = True

    should = await telegram_service._should_respond(event)
    assert should is True


@pytest.mark.asyncio
async def test_should_respond_false_in_group_chat_without_command(telegram_service):
    """Verify silence in group chat when not mentioned/replied."""
    event = AsyncMock()
    event.raw_text = "hola amigos"
    event.reply_to_msg_id = None
    event.is_private = False

    should = await telegram_service._should_respond(event)
    assert should is False


@pytest.mark.asyncio
async def test_build_recent_history_rolling_window(telegram_service):
    """Verify fetching of recent history in chronological order."""
    event = AsyncMock()
    event.chat_id = 100
    event.id = 500

    # Mock messages returned by client.get_messages
    # They come newest first usually
    # Explicitly set photo=None, id, and reply_to_msg_id to prevent MagicMock auto-generation
    msg1 = MagicMock(
        raw_text="oldest",
        sender_id=999,
        media=None,
        photo=None,
        id=497,
        reply_to_msg_id=None,
    )  # User
    msg2 = MagicMock(
        raw_text="middle",
        sender_id=12345,
        media=None,
        photo=None,
        id=498,
        reply_to_msg_id=497,
    )  # Bot
    msg3 = MagicMock(
        raw_text="newest",
        sender_id=999,
        media=None,
        photo=None,
        id=499,
        reply_to_msg_id=498,
    )  # User

    telegram_service.bot.get_messages.return_value = [msg3, msg2, msg1]

    # Patch _collect_image_attachments to avoid awaiting on MagicMock.download_media
    telegram_service._collect_image_attachments = AsyncMock(return_value=[])

    history = await telegram_service._build_recent_history(event, max_count=3)

    assert len(history) == 3
    # Check order - function should reverse them to be chronological (oldest -> newest)
    assert history[0]["content"] == "oldest"
    assert history[0]["role"] == "user"

    assert history[1]["content"] == "middle"
    assert history[1]["role"] == "assistant"

    assert history[2]["content"] == "newest"
    assert history[2]["role"] == "user"


@pytest.mark.asyncio
async def test_build_recent_history_filtered_cache_empty_falls_back_to_api(
    mock_neibot, mock_client
):
    """
    When cached messages exist but all are filtered out (message_id >= current_msg_id),
    the method should fall back to Telegram API instead of returning empty history.
    """
    # Create a mock chat repository with cached messages that will be filtered out
    mock_chat_repo = MagicMock()
    # Cached messages have message_id >= current_msg_id (500), so all will be filtered
    mock_chat_repo.get_recent_messages.return_value = [
        {"message_id": 500, "payload": {"role": "user", "content": "same id"}},
        {"message_id": 501, "payload": {"role": "user", "content": "newer"}},
        {"message_id": 502, "payload": {"role": "assistant", "content": "even newer"}},
    ]
    mock_chat_repo.save_message = MagicMock()

    mock_logger = MagicMock()

    service = TelegramService(
        command_prefix="/yo",
        neibot=mock_neibot,
        telegram_client=mock_client,
        logger=mock_logger,
        chat_repository=mock_chat_repo,
        user_service=MagicMock(),
        admin_ids=[999],
    )
    # Don't call start() - it blocks on run_until_disconnected()
    # Set self.me directly from mock_client
    service.me = await mock_client.get_me()

    event = AsyncMock()
    event.chat_id = 100
    event.id = 500  # Current message id

    # Mock Telegram API messages (fallback)
    # Explicitly set photo=None, id, and reply_to_msg_id to prevent MagicMock auto-generation
    msg1 = MagicMock(
        raw_text="from api oldest",
        sender_id=999,
        media=None,
        photo=None,
        id=498,
        reply_to_msg_id=None,
    )
    msg2 = MagicMock(
        raw_text="from api newest",
        sender_id=12345,
        media=None,
        photo=None,
        id=499,
        reply_to_msg_id=498,
    )
    service.bot.get_messages.return_value = [msg2, msg1]

    # Patch _collect_image_attachments to avoid awaiting on MagicMock.download_media
    service._collect_image_attachments = AsyncMock(return_value=[])

    history = await service._build_recent_history(event, max_count=3)

    # Should have called Telegram API since filtered cache was empty
    service.bot.get_messages.assert_called_once_with(100, limit=3, max_id=500)

    # Should have messages from API
    assert len(history) == 2
    assert history[0]["content"] == "from api oldest"
    assert history[1]["content"] == "from api newest"

    # Verify debug log was called for empty filtered cache
    mock_logger.debug.assert_any_call(
        "Filtered cache is empty for current_msg_id=%s, "
        "falling back to Telegram API",
        500,
    )


@pytest.mark.asyncio
async def test_cache_saves_only_current_attachments_not_merged_history(
    mock_neibot, mock_client
):
    """
    When caching user messages in private chats, only the current message's
    attachments should be saved, NOT the merged history attachments.
    This prevents attachment duplication when loading from cache.

    This test verifies that the _merge_attachment_lists result is used for LLM context
    but the original current_attachments are used when saving to cache.
    """
    mock_chat_repo = MagicMock()
    # Return cached history with attachments
    mock_chat_repo.get_recent_messages.return_value = [
        {
            "message_id": 98,
            "payload": {
                "role": "user",
                "content": "check this image",
                "attachments": [{"type": "image", "data": "historical_image_base64"}],
            },
        },
        {
            "message_id": 99,
            "payload": {
                "role": "assistant",
                "content": "I see the image",
                "attachments": [],
            },
        },
    ]
    mock_chat_repo.save_message = MagicMock()

    mock_user_service = MagicMock()
    mock_user_service.is_user_pro.return_value = True
    mock_user_service.get_user_max_history_turns.return_value = 20
    mock_user_service.get_user_model.return_value = "default-model"

    mock_logger = MagicMock()

    service = TelegramService(
        command_prefix="/yo",
        neibot=mock_neibot,
        telegram_client=mock_client,
        logger=mock_logger,
        chat_repository=mock_chat_repo,
        user_service=mock_user_service,
        admin_ids=[999],
    )
    # Don't call start() - it blocks on run_until_disconnected()
    # Set self.me directly from mock_client
    service.me = await mock_client.get_me()

    # Create event for a private chat message WITHOUT attachments
    event = AsyncMock()
    event.chat_id = 100
    event.id = 100
    event.sender_id = 999
    event.raw_text = "/yo what about now?"
    event.is_private = True
    event.reply_to_msg_id = None
    event.message = MagicMock(media=None)  # No media in current message

    # Mock internal methods to isolate the test
    service._should_respond = AsyncMock(return_value=True)
    service._collect_image_attachments = AsyncMock(
        return_value=[]
    )  # No current attachments
    service._is_private_chat = AsyncMock(return_value=True)

    # Mock the private __build_metadata method
    service._TelegramService__build_metadata = AsyncMock(return_value="[User: test]")

    # Mock reply to return a sent message
    sent_msg = AsyncMock()
    sent_msg.id = 101
    event.reply = AsyncMock(return_value=sent_msg)

    # Mock the typing context manager
    service.bot.action = MagicMock()
    service.bot.action.return_value.__aenter__ = AsyncMock()
    service.bot.action.return_value.__aexit__ = AsyncMock()

    await service._my_event_handler(event)

    # Verify save_message was called for user message
    save_calls = mock_chat_repo.save_message.call_args_list

    assert len(save_calls) >= 1, "save_message should have been called at least once"

    # First call should be for user message
    user_save_call = save_calls[0]
    user_payload = user_save_call[0][2]  # Third positional arg is the payload

    # User payload should have EMPTY attachments (current message had none)
    # NOT the historical attachments from cached messages
    assert user_payload["attachments"] == [], (
        f"Expected empty attachments for current message, "
        f"but got: {user_payload['attachments']}"
    )

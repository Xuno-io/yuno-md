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
    await service.start()  # To set self.me
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
    event = AsyncMock()
    # Simulate is_private raising an exception
    type(event).is_private = property(
        lambda self: (_ for _ in ()).throw(Exception("test error"))
    )

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
    msg1 = MagicMock(raw_text="oldest", sender_id=999, media=None)  # User
    msg2 = MagicMock(raw_text="middle", sender_id=12345, media=None)  # Bot (me)
    msg3 = MagicMock(raw_text="newest", sender_id=999, media=None)  # User

    telegram_service.bot.get_messages.return_value = [msg3, msg2, msg1]

    history = await telegram_service._build_recent_history(event, max_count=3)

    assert len(history) == 3
    # Check order - function should reverse them to be chronological (oldest -> newest)
    assert history[0]["content"] == "oldest"
    assert history[0]["role"] == "user"

    assert history[1]["content"] == "middle"
    assert history[1]["role"] == "assistant"

    assert history[2]["content"] == "newest"
    assert history[2]["role"] == "user"

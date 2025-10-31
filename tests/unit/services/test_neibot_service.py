import logging
from types import SimpleNamespace

import pytest

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service import NeibotService


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class DummyOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.last_messages = None

    def _create(self, **kwargs):
        self.last_messages = kwargs.get("messages")
        return DummyResponse("ok")


@pytest.fixture
def openai_client() -> DummyOpenAIClient:
    return DummyOpenAIClient()


@pytest.fixture
def neibot_service(openai_client: DummyOpenAIClient) -> NeibotService:
    return NeibotService(
        system_prompt="system context",
        model_name="gpt-test",
        openai_client=openai_client,
        logger=logging.getLogger("NeibotServiceTest"),
    )


def test_get_response_builds_multimodal_messages(
    neibot_service: NeibotService, openai_client: DummyOpenAIClient
) -> None:
    history: list[MessagePayload] = [
        {
            "role": "user",
            "content": "Hola",
            "attachments": [
                {
                    "mime_type": "image/png",
                    "size_bytes": 10,
                    "base64": "ZmFrZWltYWdl",
                    "file_name": "foto.png",
                }
            ],
        }
    ]

    result = neibot_service.get_response(history)

    assert result == "ok"
    assert openai_client.last_messages is not None
    assert openai_client.last_messages[0]["role"] == "system"

    user_message = openai_client.last_messages[1]
    assert user_message["role"] == "user"
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,ZmFrZWltYWdl"},
    }

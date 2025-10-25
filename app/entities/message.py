from typing import TypedDict, Literal


class ImageAttachment(TypedDict):
    """Binary image payload encoded as base64."""

    base64: str
    file_name: str | None
    mime_type: str
    size_bytes: int


class MessagePayload(TypedDict):
    """Conversation message used to build the LLM prompt."""

    attachments: list[ImageAttachment]
    content: str
    role: Literal["system", "user", "assistant"]

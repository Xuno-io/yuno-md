from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from app.entities.message import MessagePayload, ImageAttachment
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface


class NeibotService(NeibotServiceInterface):
    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        openai_client: OpenAI,
        logger: logging.Logger,
    ) -> None:
        self.openai_client: OpenAI = openai_client
        self.logger: logging.Logger = logger
        self.model_name: str = model_name
        self.system_prompt: str = system_prompt

        self.logger.info(f"NeibotService initialized with model: {self.model_name}")
        self.logger.info(f"System prompt: {self.system_prompt}")

    def get_response(self, history: list[MessagePayload]) -> str:
        conversation: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                    }
                ],
            }
        ]

        for message in history:
            content_parts: list[dict[str, Any]] = []

            if message["content"]:
                content_parts.append({"type": "text", "text": message["content"]})

            for attachment in message["attachments"]:
                content_parts.append(self._build_image_part(attachment))

            if not content_parts:
                content_parts.append({"type": "text", "text": ""})

            conversation.append({"role": message["role"], "content": content_parts})

        # print(conversation)
        try:
            response: ChatCompletion = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                n=1,
                stop=None,
                temperature=0.7,
            )
            message_content = response.choices[0].message.content
            if isinstance(message_content, str):
                return message_content.strip()

            return "".join(part.get("text", "") for part in message_content).strip()
        except Exception as e:
            self.logger.error(f"Error getting response from OpenAI: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

    @staticmethod
    def _build_image_part(attachment: ImageAttachment) -> dict[str, Any]:
        # The tests and the expected multimodal message format require an
        # "input_image" part with the base64 payload and mime type as fields.
        # Keep the shape as simple dict fields so unit tests and any
        # downstream consumer relying on this structure continue to work.
        return {
            "type": "input_image",
            "image_base64": attachment.get("base64"),
            "mime_type": attachment.get("mime_type"),
        }

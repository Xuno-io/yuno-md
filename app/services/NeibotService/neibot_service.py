from __future__ import annotations

import logging
from typing import Any, List, Dict

import tiktoken
from langfuse.openai import AsyncOpenAI
from langfuse import observe
from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface


class NeibotService(NeibotServiceInterface):
    """
    Service for Neibot using direct OpenAI/LiteLLM API calls with Langfuse tracing.
    Replaces the DSPy implementation to allow proper caching and native message structure.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float,
        max_tokens: int,
        logger: logging.Logger,
        cache_threshold: int = 2048,
    ) -> None:
        """
        Initialize the service with OpenAI configuration.

        Args:
            system_prompt: System instructions and personality
            model_name: Default model name
            api_key: OpenAI API key
            api_base: OpenAI API base URL
            temperature: Model temperature
            max_tokens: Max tokens for generation
            logger: Logger instance
            cache_threshold: Token threshold to activate caching (default: 2048)
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.cache_threshold = cache_threshold

        # Initialize Langfuse-wrapped OpenAI client for automatic tracing
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logger.info(
            "NeibotService initialized with direct OpenAI client (Langfuse) and threshold %s",
            self.cache_threshold,
        )

    @observe()
    async def get_response(
        self, history: list[MessagePayload], model_name: str | None = None
    ) -> str:
        """
        Get a response using OpenAI Chat Completions API.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default

        Returns:
            The assistant's response
        """
        try:
            messages = self._build_messages(history)

            # Use override model if provided, else default
            current_model = model_name or self.model_name

            if current_model != self.model_name:
                self.logger.info(f"Using custom model {current_model}")

            response = await self.client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            self.logger.error(f"Error getting response from OpenAI: {e}", exc_info=True)
            return "I'm sorry, I couldn't process your request at the moment."

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages using tiktoken.
        Only counts text content for simplicity and speed.
        """
        count = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                count += len(self.tokenizer.encode(content))
            elif isinstance(content, list):
                # Handle multimodal content list
                for part in content:
                    if part.get("type") == "text":
                        count += len(self.tokenizer.encode(part.get("text", "")))
        return count

    def _build_messages(self, history: list[MessagePayload]) -> list[dict[str, Any]]:
        """
        Convert internal MessagePayload history to OpenAI message format.
        Handles text and image attachments.
        Implements smart caching based on token threshold.
        """
        messages: list[dict[str, Any]] = []

        # Always prepend the system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in history:
            role = msg.get("role", "user")
            content_text = msg.get("content", "")
            attachments = msg.get("attachments", [])

            if not attachments:
                # Simple text message
                messages.append({"role": role, "content": content_text})
            else:
                # Multimodal message
                content_parts: list[dict[str, Any]] = []

                # Add text part if exists
                if content_text:
                    content_parts.append({"type": "text", "text": content_text})

                # Add images
                for attachment in attachments:
                    base64_data = attachment["base64"]
                    mime_type = attachment.get("mime_type", "image/jpeg")

                    # Construct data URL
                    image_url = f"data:{mime_type};base64,{base64_data}"

                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                            # Images in history could theoretically be cached,
                            # but we control cache via the message level flag.
                        }
                    )

                messages.append({"role": role, "content": content_parts})

        # Smart Caching Logic
        # We want to cache the "floor" (static history) if it exceeds the threshold.
        # The floor is everything EXCEPT the last message (which is the current user query).

        if len(messages) > 1:
            # Candidates for caching: all messages except the last one
            static_context = messages[:-1]

            total_tokens = self._count_tokens(static_context)

            if total_tokens > self.cache_threshold:
                # Mark the last message of the static context as the cache checkpoint
                # This tells the provider to cache everything up to this point
                messages[-2]["cache_control"] = {"type": "ephemeral"}
                self.logger.info(
                    "Smart Caching ACTIVATED: %s tokens in static context (Threshold: %s)",
                    total_tokens,
                    self.cache_threshold,
                )
            else:
                self.logger.debug(
                    "Smart Caching SKIPPED: %s tokens (Threshold: %s)",
                    total_tokens,
                    self.cache_threshold,
                )

        return messages

from __future__ import annotations

import asyncio
import logging

import dspy

from app.entities.message import MessagePayload, ImageAttachment
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.NeibotService.dspy_signatures import (
    ConversationSignature,
    MessageSignature,
)
import base64
from langfuse import observe
from PIL import Image
from io import BytesIO


class NeibotDSPyService(NeibotServiceInterface):
    """DSPy-powered service for Neibot using YunoAI with Langfuse tracing."""

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        api_key: str,
        api_base: str,
        temperature: float,
        max_tokens: int,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the DSPy service with YunoAI configuration.

        Args:
            system_prompt: System instructions and personality
            model_name: Default model name
            api_key: OpenAI API key
            api_base: OpenAI API base URL
            temperature: Model temperature
            max_tokens: Max tokens for generation
            logger: Logger instance
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        # Initialize default LM
        self.default_lm = self._create_lm(self.model_name)

        # Initialize DSPy modules with default LM
        dspy.configure(lm=self.default_lm)
        self.conversation_module = dspy.ChainOfThought(ConversationSignature)

        self.logger.info("NeibotDSPyService initialized with DSPy")

    def _create_lm(self, model: str) -> dspy.LM:
        return dspy.LM(
            model=model,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @observe()
    async def get_response(
        self, history: list[MessagePayload], model_name: str | None = None
    ) -> str:
        """
        Get a response using DSPy with conversation context.

        Implements the NeibotServiceInterface method to maintain compatibility.
        Supports images through MessagePayload format.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default

        Returns:
            The assistant's response
        """
        try:
            # Determine which LM to use
            current_lm = self.default_lm
            if model_name and model_name != self.model_name:
                self.logger.info(f"Using custom model {model_name}")
                current_lm = self._create_lm(model_name)

            # Build context from history (includes text and image descriptions)
            # Exclude the last message as it is passed separately as 'question'
            context = await self._build_context(history[:-1])

            # Get the last message as the current question
            current_message = history[-1] if history else None
            question = current_message.get("content", "") if current_message else ""

            # Add attachments from the current message to context
            # We excluded the text above (to be 'question'), but we need the images in 'context'
            if current_message and current_message.get("attachments"):
                for attachment in current_message["attachments"]:
                    base64_data = attachment["base64"]
                    pil_image = Image.open(BytesIO(base64.b64decode(base64_data)))
                    image = dspy.Image.from_PIL(pil_image)
                    context_line = MessageSignature(content=image)
                    context.append(context_line)

            # Use DSPy to generate response
            # Execute the synchronous DSPy call in a thread pool to avoid blocking the event loop
            # This allows the typing indicator to remain active during processing
            # Use dspy.context to apply the selected LM
            def run_dspy():
                with dspy.context(lm=current_lm):
                    return self.conversation_module(
                        system_prompt=self.system_prompt,
                        context=context,
                        question=question,
                    )

            result = await asyncio.to_thread(run_dspy)

            # DSPy returns a Prediction object with the output fields
            return result.answer.strip()

        except Exception as e:
            self.logger.error(f"Error getting DSPy response: {e}", exc_info=True)
            return "I'm sorry, I couldn't process your request at the moment."

    async def _build_context(
        self, history: list[MessagePayload]
    ) -> list[MessageSignature]:
        """
        Build conversation context from history.

        Handles both text content and image attachments by creating MessageSignature
        objects for DSPy processing.

        Args:
            history: List of MessagePayload objects

        Returns:
            List of MessageSignature objects representing the conversation history
        """
        if not history:
            return []

        context_lines = []

        # Process all messages except the last one (which is the current question)
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            attachments: list[ImageAttachment] = msg.get("attachments", [])

            # Build message line
            if content:
                context_line = MessageSignature(content=f"{role}: {content}")
                context_lines.append(context_line)

            # Add image information to context
            if attachments:
                for attachment in attachments:
                    # Use direct access since ImageAttachment TypedDict guarantees base64 exists
                    base64_data = attachment["base64"]
                    pil_image = Image.open(BytesIO(base64.b64decode(base64_data)))
                    image = dspy.Image.from_PIL(pil_image)
                    context_line = MessageSignature(content=image)
                    context_lines.append(context_line)

        return context_lines

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
        lm: dspy.LM,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the DSPy service with YunoAI configuration.

        Args:
            system_prompt: System instructions and personality
            lm: Configured DSPy LM instance
            logger: Logger instance
        """
        self.system_prompt = system_prompt
        self.logger = logger
        self.lm = lm

        # Initialize DSPy modules
        dspy.configure(lm=self.lm)
        self.conversation_module = dspy.ChainOfThought(ConversationSignature)

        self.logger.info("NeibotDSPyService initialized with DSPy")

    @observe()
    async def get_response(self, history: list[MessagePayload]) -> str:
        """
        Get a response using DSPy with conversation context.

        Implements the NeibotServiceInterface method to maintain compatibility.
        Supports images through MessagePayload format.

        Args:
            history: List of MessagePayload objects with role, content, and attachments

        Returns:
            The assistant's response
        """
        try:
            # Build context from history (includes text and image descriptions)
            context = await self._build_context(history)

            # Get the last message as the current question
            current_message = history[-1] if history else None
            question = current_message.get("content", "") if current_message else ""

            # Use DSPy to generate response
            # Execute the synchronous DSPy call in a thread pool to avoid blocking the event loop
            # This allows the typing indicator to remain active during processing
            result = await asyncio.to_thread(
                self.conversation_module,
                system_prompt=self.system_prompt,
                context=context,
                question=question,
            )

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

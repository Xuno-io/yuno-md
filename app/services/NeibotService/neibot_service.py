from __future__ import annotations

import logging
import base64

from google import genai
from google.genai import types
from langfuse import observe

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface


class NeibotService(NeibotServiceInterface):
    """
    Service for Neibot using Google Gen AI SDK (v2) with Langfuse tracing.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str,
        location: str,
        project_id: str | None,
        temperature: float,
        max_tokens: int,
        logger: logging.Logger,
        cache_threshold: int = 2048,
    ) -> None:
        """
        Initialize the service with Google Gen AI configuration.

        Args:
            system_prompt: System instructions and personality
            model_name: Default model name
            location: Vertex AI location (e.g., "us-central1")
            project_id: GCP Project ID (optional)
            temperature: Model temperature
            max_tokens: Max tokens for generation
            logger: Logger instance
            cache_threshold: Token threshold (unused in this version, kept for interface)
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.location = location
        self.project_id = project_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.cache_threshold = cache_threshold

        # Initialize Google Gen AI Client (Vertex AI backend)
        self.client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )

        self.logger.info(
            "NeibotService initialized with Google Gen AI SDK. Model: %s, Location: %s",
            self.model_name,
            self.location,
        )

    @observe()
    async def get_response(
        self, history: list[MessagePayload], model_name: str | None = None
    ) -> str:
        """
        Get a response using Google Gen AI SDK.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default

        Returns:
            The assistant's response
        """
        try:
            contents = self._build_contents(history)

            # Use override model if provided
            current_model = model_name or self.model_name
            if current_model != self.model_name:
                self.logger.info(f"Using custom model {current_model}")

            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=self.system_prompt,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )

            response = await self.client.aio.models.generate_content(
                model=current_model,
                contents=contents,
                config=config,
            )

            if not response:
                self.logger.warning("Received empty response from Google Gen AI")
                return ""

            # Google Gen AI response handling
            if response.text:
                return response.text

            # Check for safety blocking if no text
            if (
                response.candidates
                and response.candidates[0].finish_reason
                == "SAFETY"  # Check constant value if available, using string literal for robustness
            ):
                self.logger.warning(
                    f"Response blocked due to safety: {response.candidates[0].safety_ratings}"
                )
                return (
                    "I'm sorry, I couldn't process your request due to safety filters."
                )

            return ""

        except Exception as e:
            self.logger.error(
                f"Error getting response from Google Gen AI: {e}", exc_info=True
            )
            return "I'm sorry, I couldn't process your request at the moment."

    def _build_contents(self, history: list[MessagePayload]) -> list[types.Content]:
        """
        Convert internal MessagePayload history to Google Gen AI Content objects.
        """
        contents: list[types.Content] = []

        for msg in history:
            role = msg.get("role", "user")

            # Map roles to Google Gen AI (user, model)
            # System prompt is handled in config, skip system messages here
            if role == "system":
                continue
            elif role == "assistant":
                genai_role = "model"
            else:
                genai_role = "user"

            content_text = msg.get("content", "")
            attachments = msg.get("attachments", [])

            parts: list[types.Part] = []

            # Add text part
            if content_text:
                parts.append(types.Part.from_text(text=content_text))

            # Add images
            for attachment in attachments:
                base64_data = attachment.get("base64")
                mime_type = attachment.get("mime_type", "image/jpeg")

                if base64_data and isinstance(base64_data, str):
                    try:
                        image_bytes = base64.b64decode(base64_data)
                        parts.append(
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to decode attachment: {e}")
                        continue
                else:
                    self.logger.warning(
                        "Attachment missing base64 data or invalid format"
                    )
                    continue

            if parts:
                contents.append(types.Content(role=genai_role, parts=parts))

        return contents

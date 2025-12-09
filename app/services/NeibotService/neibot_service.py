from __future__ import annotations

import logging
import base64
import json
from datetime import datetime, timezone

from google import genai
from google.genai import types
from langfuse import observe

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.KnowledgeService.knowledge_service_interface import (
    KnowledgeServiceInterface,
)


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
        knowledge_service: KnowledgeServiceInterface | None = None,
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
            knowledge_service: Optional service for accessing the knowledge base
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.location = location
        self.project_id = project_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.cache_threshold = cache_threshold
        self.knowledge_service = knowledge_service

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
        self,
        history: list[MessagePayload],
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Get a response using Google Gen AI SDK.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default
            user_id: Optional user ID to fetch knowledge context

        Returns:
            The assistant's response
        """
        try:
            contents = self._build_contents(history)

            # Use override model if provided
            current_model = model_name or self.model_name
            if current_model != self.model_name:
                self.logger.info(f"Using custom model {current_model}")

            # Fetch knowledge context if user_id is provided
            knowledge_context = ""
            if user_id and self.knowledge_service:
                try:
                    # Query all active facts for this entity
                    facts = self.knowledge_service.query(
                        {"entity": user_id, "active_only": True}
                    )
                    if facts:
                        facts_list = [
                            f"- {f.get('attribute')}: {f.get('value')}" for f in facts
                        ]
                        knowledge_context = "\n\n[MEMORY CONTEXT]\n" + "\n".join(
                            facts_list
                        )
                        self.logger.info(
                            f"Injected {len(facts)} facts for user {user_id}"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to fetch knowledge context: {e}")

            # Add current UTC time to system prompt
            utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            system_prompt_with_time = f"{self.system_prompt}\n\nCurrent time (UTC): {utc_time}{knowledge_context}"

            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=system_prompt_with_time,
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

    @observe()
    async def capture_facts_from_history(
        self,
        history: list[MessagePayload],
        user_id: str,
    ) -> int:
        """
        Analyze history, extract facts, and save them to the knowledge base.
        Returns the number of facts saved.
        """
        if not self.knowledge_service:
            self.logger.warning("Knowledge service not available for capturing facts.")
            return 0

        try:
            # Prepare conversation text
            conversation_text = ""
            for msg in history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                conversation_text += f"{role}: {content}\n"

            prompt = (
                f"Analyze the following conversation history and extract stable facts about the USER (id: {user_id}).\n"
                "Ignore trivial details, greetings, or temporary context.\n"
                "Focus on: names, preferences, job, location, interests, specific relationships.\n"
                "Return the output as a JSON list of objects. Each object must have:\n"
                "  - 'attribute': string (snake_case key, e.g., 'user_name', 'favorite_color')\n"
                "  - 'value': string, number, or boolean (the fact value)\n"
                "  - 'confidence': number (0.0 to 1.0)\n"
                "If no facts are found, return an empty list [].\n\n"
                "Conversation:\n"
                f"{conversation_text}"
            )

            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=config,
            )

            if not response.text:
                return 0

            facts = json.loads(response.text)
            if not isinstance(facts, list):
                return 0

            saved_count = 0
            for fact in facts:
                if not isinstance(fact, dict) or fact.get("confidence", 0) <= 0.7:
                    continue

                attr = fact.get("attribute")
                val = fact.get("value")
                if attr and val is not None:
                    if self.knowledge_service.add_fact(user_id, attr, val):
                        saved_count += 1

            return saved_count

        except Exception as e:
            self.logger.error(f"Error capturing facts: {e}", exc_info=True)
            return 0

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

"""
NeibotService using Google Agent Development Kit (ADK).

This service uses the ADK framework for conversational agent execution,
replacing the manual tool-calling loop with ADK's automatic orchestration.
"""

from __future__ import annotations

import logging
import asyncio
import base64
import uuid
import contextvars
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from google import genai
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.events import Event
from langfuse import observe

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.MemoryService.memory_service_interface import MemoryServiceInterface
from app.tools.web_search_tool import web_search

if TYPE_CHECKING:
    pass


# Context variable for thread-safe user context access
_current_user_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id", default=None
)


# Default distillation prompt in case the file is not found
DEFAULT_DISTILL_PROMPT = """Sistema: Estás operando en un estado de excepción por saturación de flujo (Límite de caracteres).
Tu misión es destilar la intensidad de tu respuesta fallida.
Reconstruye el mensaje original en CUATRO movimientos obligatorios.

1. EL ESTRATO (El Juez): Expón el núcleo duro de la idea.
2. EL AGRIETAMIENTO (El Traidor): Encuentra el punto débil.
3. LA LÍNEA DE FUGA (El Visionario): ¿Qué nueva forma emerge?
4. EL VECTOR (El Mecánico): Define UN solo paso accionable.

RESPUESTA ORIGINAL A DESTILAR:
"""


# Application name for ADK sessions
ADK_APP_NAME = "yunoai"


class NeibotService(NeibotServiceInterface):
    """
    Service for Neibot using Google Agent Development Kit (ADK) with Langfuse tracing.

    ADK handles the tool-calling loop automatically, simplifying the code
    and improving reliability.
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
        memory_service: MemoryServiceInterface | None = None,
        extraction_model_name: str | None = None,
        distill_model_name: str | None = None,
    ) -> None:
        """
        Initialize the service with Google ADK configuration.

        Args:
            system_prompt: System instructions and personality
            model_name: Default model name for conversations
            location: Vertex AI location (e.g., "us-central1")
            project_id: GCP Project ID (optional)
            temperature: Model temperature
            max_tokens: Max tokens for generation
            logger: Logger instance
            cache_threshold: Token threshold (unused in this version, kept for interface)
            memory_service: mem0-based memory service for long-term memory
            extraction_model_name: Model for fact extraction (defaults to gemini-3-flash-preview)
            distill_model_name: Model for response distillation when messages are too long
        """
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.location = location
        self.project_id = project_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger
        self.cache_threshold = cache_threshold
        self.memory_service = memory_service
        self.extraction_model_name = extraction_model_name or "gemini-3-flash-preview"
        self.distill_model_name = distill_model_name or "gemini-2.5-pro"

        # Load the distillation prompt from file
        self.distill_prompt = self._load_distill_prompt()

        # Initialize Google Gen AI Client (Vertex AI backend) for distillation
        self.client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )

        # Initialize ADK session service (in-memory for stateless per-request usage)
        self.session_service = InMemorySessionService()

        self.logger.info(
            "NeibotService initialized with Google ADK. Model: %s, Location: %s",
            self.model_name,
            self.location,
        )

    def _create_search_memory_tool(self) -> Callable:
        """
        Create a search_memory tool function with access to the memory service.

        Returns a closure that captures self for memory service access.
        """

        async def search_memory(query: str, category: str) -> dict:
            """
            Search long-term memory for project-specific facts about this user.

            Use ONLY when the user asks about their specific project details,
            past decisions, or constraints. DO NOT use for greetings or general knowledge.

            Args:
                query: The specific question or topic to search for
                category: The memory category to search in.
                    Options: TECH_STACK (for code/infra),
                            BUSINESS_LOGIC (for rules/budgets),
                            USER_CONSTRAINTS (for limitations)

            Returns:
                Dictionary with search results or error message
            """
            user_id = _current_user_context.get()

            if not self.memory_service:
                return {"status": "error", "result": "Memory service not available."}

            if not user_id:
                return {"status": "error", "result": "No user context available."}

            self.logger.info(
                "Searching memory for user %s: query='%s', category=%s",
                user_id,
                query,
                category,
            )

            try:
                memories = self.memory_service.search(
                    query=query,
                    user_id=user_id,
                    category=category,
                    limit=5,
                )

                if not memories:
                    return {
                        "status": "success",
                        "result": "No relevant memories found for this query.",
                    }

                # Format memories for the LLM
                from app.services.MemoryService.memory_service import (
                    parse_embedded_category,
                )

                formatted = []
                for mem in memories:
                    text = mem.get("memory") or mem.get("text", "")
                    cat, clean_text = parse_embedded_category(text)
                    formatted.append(f"[{cat}] {clean_text}")

                result = "\n".join(formatted)
                self.logger.info(
                    "Found %d memories for query '%s'", len(memories), query
                )
                return {"status": "success", "result": result}

            except Exception as e:
                self.logger.error("Memory search failed: %s", e)
                return {"status": "error", "result": f"Memory search failed: {str(e)}"}

        return search_memory

    def _create_agent(self, model_name: str) -> Agent:
        """
        Create an ADK Agent with the configured tools.

        Args:
            model_name: The model to use for this agent instance

        Returns:
            Configured ADK Agent
        """
        # Build tools list
        tools: list[Callable[..., Any] | Any] = [web_search]

        # Add memory search if memory service is available
        if self.memory_service:
            tools.append(self._create_search_memory_tool())

        # Add current UTC time to system prompt
        utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        instruction_with_time = (
            f"{self.system_prompt}\n\nCurrent time (UTC): {utc_time}"
        )

        return Agent(
            model=model_name,
            name="yuno_agent",
            description="YunoAI - An intellectual catalyst for founders and creators",
            instruction=instruction_with_time,
            tools=tools,
        )

    async def _create_session_with_history(
        self,
        user_id: str,
        history: list[MessagePayload],
    ) -> Session:
        """
        Create an ADK session and pre-populate it with conversation history.

        Args:
            user_id: User identifier
            history: List of previous messages (excluding the latest user message)

        Returns:
            Session with history loaded
        """
        session_id = str(uuid.uuid4())

        # Create a new session
        session = await self.session_service.create_session(
            app_name=ADK_APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        # Pre-populate history by appending events
        # Skip the last message as it will be sent as the new_message
        history_to_load = history[:-1] if history else []

        for msg in history_to_load:
            role = msg.get("role", "user")
            content_text = msg.get("content", "")

            # Skip system messages (handled in agent instruction)
            if role == "system" or not content_text:
                continue

            # Map roles: user -> user, assistant -> agent
            author = "user" if role == "user" else "agent"

            # Create content with text
            parts = [types.Part.from_text(text=content_text)]

            # Handle attachments for user messages
            if role == "user":
                attachments = msg.get("attachments", [])
                for attachment in attachments:
                    base64_data = attachment.get("base64")
                    mime_type = attachment.get("mime_type", "image/jpeg")
                    if base64_data and isinstance(base64_data, str):
                        try:
                            image_bytes = base64.b64decode(base64_data)
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes, mime_type=mime_type
                                )
                            )
                        except Exception as e:
                            self.logger.warning("Failed to decode attachment: %s", e)

            content = types.Content(role=author, parts=parts)

            # Create and append event
            event = Event(
                author=author,
                content=content,
            )
            await self.session_service.append_event(session=session, event=event)

        return session

    @observe()
    async def get_response(
        self,
        history: list[MessagePayload],
        model_name: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """
        Get a response using Google ADK with automatic tool orchestration.

        The ADK Agent automatically handles tool calls (search_memory, web_search)
        and manages the conversation loop.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default
            user_id: Optional user ID for memory operations

        Returns:
            The assistant's response
        """
        token = None
        try:
            if not history:
                self.logger.warning("Empty history provided")
                return ""

            # Store user_id for tool access using contextvars
            token = _current_user_context.set(user_id)

            # Use override model if provided
            current_model = model_name or self.model_name
            if current_model != self.model_name:
                self.logger.info("Using custom model %s", current_model)

            # Create agent with current model
            agent = self._create_agent(current_model)

            # Create runner
            runner = Runner(
                agent=agent,
                app_name=ADK_APP_NAME,
                session_service=self.session_service,
            )

            # Create session with conversation history
            effective_user_id = user_id or "anonymous"
            session = await self._create_session_with_history(
                user_id=effective_user_id,
                history=history,
            )

            # Get the latest message to send
            last_message = history[-1]
            last_content = last_message.get("content", "")
            attachments = last_message.get("attachments", [])

            # Build the new message content
            parts = [types.Part.from_text(text=last_content)] if last_content else []

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
                        self.logger.warning("Failed to decode attachment: %s", e)

            new_message = types.Content(role="user", parts=parts)

            # Check for empty content to avoid API errors
            if not new_message.parts:
                self.logger.warning(
                    "new_message parts empty (no valid content or attachments). Using safe default."
                )
                new_message.parts = [
                    types.Part.from_text(
                        text="[System note: User sent an empty message or unsupported attachment]"
                    )
                ]

            # Run the agent and collect the final response
            final_response = ""

            try:
                # Set a timeout to prevent indefinite hanging (e.g. tool loop issues)
                async with asyncio.timeout(300):
                    async for event in runner.run_async(
                        user_id=effective_user_id,
                        session_id=session.id,
                        new_message=new_message,
                    ):
                        # Check for final response
                        if event.is_final_response():
                            if event.content and event.content.parts:
                                for part in event.content.parts:
                                    if hasattr(part, "text") and part.text:
                                        final_response += part.text

            except TimeoutError:
                self.logger.error("ADK runner execution timed out")
                return "I apologize, but this request is taking longer than expected. Please try again."

            if not final_response:
                self.logger.warning("ADK returned empty response")
                return ""

            return final_response

        except Exception as e:
            self.logger.error("Error getting response from ADK: %s", e, exc_info=True)
            return "I'm sorry, I couldn't process your request at the moment."
        finally:
            if token:
                _current_user_context.reset(token)

    @observe()
    async def capture_facts_from_history(
        self,
        history: list[MessagePayload],
        user_id: str,
    ) -> int:
        """
        Analyze conversation history and save memories using mem0.

        Returns the number of memories processed.
        """
        if not self.memory_service:
            self.logger.warning("No memory service available.")
            return 0
        return await self._capture_with_mem0(history, user_id)

    async def _capture_with_mem0(
        self,
        history: list[MessagePayload],
        user_id: str,
    ) -> int:
        """
        Capture memories using mem0 (new system).

        mem0's custom_prompt handles:
        - Filtering trivial facts
        - Categorizing into TECH_STACK, BUSINESS_LOGIC, USER_CONSTRAINTS
        """
        try:
            # Build conversation text
            conversation_text = ""
            for msg in history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                conversation_text += f"{role}: {content}\n"

            if not conversation_text.strip():
                return 0

            if self.memory_service is None:
                return 0
            result = self.memory_service.add(
                content=conversation_text,
                user_id=user_id,
            )

            # Count memories created - use only the standardized "results" key
            memories_created = 0
            if isinstance(result, dict):
                results = result.get("results", [])
                memories_created = len(results) if isinstance(results, list) else 0

            self.logger.info(
                "mem0 processed conversation for user %s: %s memories",
                user_id,
                memories_created,
            )
            return memories_created

        except Exception as e:
            self.logger.error("Error capturing with mem0: %s", e, exc_info=True)
            return 0

    def _load_distill_prompt(self) -> str:
        """
        Load the distillation prompt from the fragment_intensively_v2.prompt file.

        Returns:
            The distillation prompt content, or a default if file is not found.
        """
        prompt_path = Path(__file__).resolve().parent / "fragment_intensively_v2.prompt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.logger.warning(
                "Distillation prompt file not found at %s, using default",
                prompt_path,
            )
            return DEFAULT_DISTILL_PROMPT
        except Exception as e:
            self.logger.error("Error loading distillation prompt: %s, using default", e)
            return DEFAULT_DISTILL_PROMPT

    @observe()
    async def distill_response(
        self,
        original_response: str,
        context: list[MessagePayload] | None = None,
    ) -> str:
        """
        Distill a long response into a condensed version using the fragmentation protocol.

        This method is called when a response exceeds Telegram's character limit.
        It applies the 4-movement distillation protocol to condense the response.

        Note: This method uses the direct GenAI client (not ADK) as it's a single-turn
        operation without tool requirements.

        Args:
            original_response: The original response that was too long
            context: Optional recent conversation history (last 10 messages) for better context

        Returns:
            A condensed version of the response following the distillation protocol
        """
        try:
            # Build context summary from recent messages (last 5 exchanges = 10 messages)
            context_summary = ""
            if context:
                # Take only last 10 messages for context
                recent_context = context[-10:]
                context_lines = []
                for msg in recent_context:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")
                    # Truncate very long messages in context to save tokens
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_lines.append(f"{role}: {content}")
                context_summary = (
                    "CONTEXTO DE LA CONVERSACIÓN:\n"
                    + "\n".join(context_lines)
                    + "\n\n---\n\n"
                )

            # Build the distillation prompt with context and original response
            full_prompt = f"{context_summary}{self.distill_prompt}\n{original_response}"

            self.logger.info(
                "Distilling response of %d characters using model %s (context: %d messages)",
                len(original_response),
                self.distill_model_name,
                len(context) if context else 0,
            )

            # Configure generation parameters for distillation
            # Use lower temperature for more focused output
            config = types.GenerateContentConfig(
                temperature=0.7,
                # Telegram limits are in characters (4096), not tokens.
                # Tokens differ from characters (typically 1 token ≈ 3-4 characters).
                # This conservative token cap (900) is chosen to keep output safely
                # under Telegram's 4096-character limit.
                max_output_tokens=2048,  # 1024 for thinking and 1024 for output
                # Allow the model to reason internally with up to 1024 tokens
                # for better distillation quality, while keeping final output ≤ 900 tokens.
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            )

            # Build content for the distillation request
            contents: list[Any] = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=full_prompt)],
                )
            ]

            # Call the LLM for distillation
            response = await self.client.aio.models.generate_content(
                model=self.distill_model_name,
                contents=contents,
                config=config,
            )

            if response and response.text:
                self.logger.info(
                    "Successfully distilled response from %d to %d characters",
                    len(original_response),
                    len(response.text),
                )
                return response.text

            self.logger.warning("Distillation returned empty response")
            return "Error: No pude destilar la respuesta. Por favor, intenta de nuevo."

        except Exception as e:
            self.logger.error("Error distilling response: %s", e, exc_info=True)
            return "Error: No pude procesar la destilación de la respuesta."

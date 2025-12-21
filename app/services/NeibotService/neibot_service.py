from __future__ import annotations

import logging
import base64
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from langfuse import observe

from app.entities.message import MessagePayload
from app.services.NeibotService.neibot_service_interface import NeibotServiceInterface
from app.services.MemoryService.memory_service_interface import MemoryServiceInterface


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


# Tool definition for memory search
SEARCH_MEMORY_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_memory",
            description=(
                "Search long-term memory for project-specific facts about this user. "
                "Use ONLY when the user asks about their specific project details, "
                "past decisions, or constraints. DO NOT use for greetings or general knowledge."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The specific question or topic to search for",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        enum=["TECH_STACK", "BUSINESS_LOGIC", "USER_CONSTRAINTS"],
                        description=(
                            "The memory category to search in. "
                            "TECH_STACK for code/infra, "
                            "BUSINESS_LOGIC for rules/budgets, "
                            "USER_CONSTRAINTS for limitations."
                        ),
                    ),
                },
                required=["query", "category"],
            ),
        )
    ]
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
        memory_service: MemoryServiceInterface | None = None,
        extraction_model_name: str | None = None,
        distill_model_name: str | None = None,
    ) -> None:
        """
        Initialize the service with Google Gen AI configuration.

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
            extraction_model_name: Model for fact extraction (defaults to gemini-2.0-flash)
            distill_model_name: Model for response distillation when messages are too long (defaults to gemini-2.5-pro)
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
        # Use a fast, cheap model for extraction tasks (structured output, no deep reasoning)
        self.extraction_model_name = extraction_model_name or "gemini-2.0-flash"
        # Model for distilling long responses (defaults to gemini-2.5-pro)
        self.distill_model_name = distill_model_name or "gemini-2.5-pro"

        # Load the distillation prompt from file
        self.distill_prompt = self._load_distill_prompt()

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
        Get a response using Google Gen AI SDK with tool execution support.

        The LLM can use the search_memory tool to query long-term memory
        when it needs context about the user's project or past decisions.

        Args:
            history: List of MessagePayload objects with role, content, and attachments
            model_name: Optional model name to override the default
            user_id: Optional user ID for memory operations

        Returns:
            The assistant's response
        """
        try:
            contents = self._build_contents(history)

            # Use override model if provided
            current_model = model_name or self.model_name
            if current_model != self.model_name:
                self.logger.info(f"Using custom model {current_model}")

            # Add current UTC time to system prompt
            utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            system_prompt_with_time = (
                f"{self.system_prompt}\n\nCurrent time (UTC): {utc_time}"
            )

            # Build tools list - use memory search if memory_service is available
            # Note: Google Search disabled - can't mix search tools with function calling
            tools_list = []
            if self.memory_service and user_id:
                tools_list.append(SEARCH_MEMORY_TOOL)

            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=system_prompt_with_time,
                tools=tools_list,
            )

            # First LLM call
            response = await self.client.aio.models.generate_content(
                model=current_model,
                contents=contents,
                config=config,
            )

            if not response:
                self.logger.warning("Received empty response from Google Gen AI")
                return ""

            # Check for function call (tool use)
            if await self._handle_tool_calls(
                response, contents, current_model, config, user_id
            ):
                # Tool was called, make second call with results
                response = await self.client.aio.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )

            # Extract final text response
            if response and response.text:
                return response.text

            # Check for safety blocking if no text
            if (
                response
                and response.candidates
                and response.candidates[0].finish_reason == "SAFETY"
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

    async def _handle_tool_calls(
        self,
        response,
        contents: list[types.Content],
        model: str,
        config: types.GenerateContentConfig,
        user_id: str | None,
    ) -> bool:
        """
        Handle function calls from the LLM response.

        Returns True if a tool was called and contents were updated.
        """
        if not response.candidates:
            return False

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return False

        tool_called = False

        for part in candidate.content.parts:
            if not hasattr(part, "function_call") or not part.function_call:
                continue

            function_call = part.function_call
            function_name = function_call.name

            self.logger.info(f"LLM requested tool: {function_name}")

            if function_name == "search_memory" and self.memory_service and user_id:
                # Execute memory search
                result = await self._execute_search_memory(function_call.args, user_id)

                # Add the function call and response to contents
                contents.append(types.Content(role="model", parts=[part]))
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name="search_memory",
                                response={"result": result},
                            )
                        ],
                    )
                )
                tool_called = True

        return tool_called

    async def _execute_search_memory(self, args: dict, user_id: str) -> str:
        """
        Execute a memory search and return formatted results.
        """
        query = args.get("query", "")
        category = args.get("category")

        self.logger.info(
            f"Searching memory for user {user_id}: query='{query}', category={category}"
        )

        try:
            if self.memory_service is None:
                return "Memory service not available."
            memories = self.memory_service.search(
                query=query,
                user_id=user_id,
                category=category,
                limit=5,
            )

            if not memories:
                return "No relevant memories found for this query."

            # Format memories for the LLM
            # Categories are embedded in the text as "[CATEGORY] text"
            from app.services.MemoryService.memory_service import (
                parse_embedded_category,
            )

            formatted = []
            for mem in memories:
                text = mem.get("memory") or mem.get("text", "")
                cat, clean_text = parse_embedded_category(text)
                formatted.append(f"[{cat}] {clean_text}")

            result = "\n".join(formatted)
            self.logger.info(f"Found {len(memories)} memories for query '{query}'")
            return result

        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return f"Memory search failed: {str(e)}"

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

            # memory_service is checked in capture_facts_from_history
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
            self.logger.error(f"Error capturing with mem0: {e}", exc_info=True)
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
                max_output_tokens=900,
                # Allow the model to reason internally with up to 1024 tokens
                # for better distillation quality, while keeping final output ≤ 900 tokens.
                # thinking_budget and max_output_tokens are independent parameters.
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            )

            # Build content for the distillation request
            contents = [
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
            self.logger.error(f"Error distilling response: {e}", exc_info=True)
            return "Error: No pude procesar la destilación de la respuesta."

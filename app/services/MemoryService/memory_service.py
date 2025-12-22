"""
Memory Service using mem0 with Redis as vector store.

Provides long-term memory capabilities with:
- Semantic search via embeddings
- Category-based filtering (TECH_STACK, BUSINESS_LOGIC, USER_CONSTRAINTS)
- Automatic fact extraction with noise filtering
"""

import logging
import os
from typing import Any

from mem0 import Memory

from app.services.MemoryService.memory_service_interface import MemoryServiceInterface


# Fact extraction prompt - filters noise and enforces embedded categorization
# Since mem0 OSS doesn't support custom_categories, we embed the category in the memory text
# IMPORTANT: mem0 expects JSON response with {"facts": [...]} format
FACT_EXTRACTION_PROMPT = """
You are a Memory Manager for a rigorous intellectual assistant.
DO NOT save trivial facts (e.g., "User said hello", "User likes blue", "User is tired").

EXTRACT ONLY facts that fall into these categories:
- TECH_STACK: Libraries, frameworks, infrastructure, code patterns, APIs, architecture
- BUSINESS_LOGIC: Rules, constraints, budgets, deadlines, processes, workflows
- USER_CONSTRAINTS: Limitations, preferences, working style that affect decisions

Each fact MUST start with the category in brackets: [CATEGORY] The actual fact

RESPOND WITH VALID JSON in this exact format:
{"facts": ["[CATEGORY] fact1", "[CATEGORY] fact2"]}

Examples of valid facts:
- "[TECH_STACK] Uses Redis for vector storage with mem0"
- "[BUSINESS_LOGIC] Maximum budget is 500 USD"
- "[USER_CONSTRAINTS] Prefers automated triggers over manual commands"

If the input is casual conversation with no strategic value, return: {"facts": []}
Do NOT include greetings, emotions, or trivial preferences.
"""

# Valid categories for parsing embedded tags
VALID_CATEGORIES = {"TECH_STACK", "BUSINESS_LOGIC", "USER_CONSTRAINTS"}


def parse_embedded_category(memory_text: str) -> tuple[str, str]:
    """
    Parse category from embedded format: "[CATEGORY] actual text"

    Returns:
        tuple: (category, clean_text) - category defaults to "OTHER" if not found
    """
    import re

    match = re.match(r"^\[([A-Z_]+)\]\s*(.+)$", memory_text, re.DOTALL)
    if match:
        category = match.group(1)
        text = match.group(2).strip()
        if category in VALID_CATEGORIES:
            return category, text
    return "OTHER", memory_text


class MemoryService(MemoryServiceInterface):
    """
    Memory service implementation using mem0 with Redis backend.

    Configuration via environment variables:
        REDIS_HOST: Redis host (default: localhost)
        REDIS_PORT: Redis port (default: 6379)
        GOOGLE_API_KEY: Required for embeddings
    """

    def __init__(
        self,
        logger: logging.Logger,
        redis_host: str | None = None,
        redis_port: int | None = None,
        llm_model: str = "gemini-3-flash-preview",
        embedder_model: str = "text-embedding-004",
    ) -> None:
        """
        Initialize the memory service with mem0 configuration.

        Args:
            logger: Logger instance
            redis_host: Redis host (defaults to env REDIS_HOST or localhost)
            redis_port: Redis port (defaults to env REDIS_PORT or 6379)
            llm_model: Model for fact extraction
            embedder_model: Model for embeddings
        """
        self.logger = logger

        # Redis configuration
        self._redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self._redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))

        # Build mem0 configuration
        # Redis uses redis_url format, not separate host/port
        redis_url = f"redis://{self._redis_host}:{self._redis_port}"

        config = {
            "vector_store": {
                "provider": "redis",
                "config": {
                    "redis_url": redis_url,
                    "collection_name": "yuno_memories",
                    "embedding_model_dims": 768,  # text-embedding-004 dimensions
                },
            },
            "llm": {
                "provider": "gemini",  # Not "google"
                "config": {
                    "model": llm_model,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "gemini",  # Not "google"
                "config": {
                    "model": embedder_model,
                },
            },
            "custom_fact_extraction_prompt": FACT_EXTRACTION_PROMPT,
        }

        try:
            self._memory = Memory.from_config(config)
            self.logger.info(
                "MemoryService initialized with Redis at %s:%s",
                self._redis_host,
                self._redis_port,
            )
        except Exception as e:
            self.logger.error("Failed to initialize mem0: %s", e)
            raise

    def add(
        self,
        content: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add a memory to the store.

        mem0 will automatically extract facts using the custom_prompt
        and categorize them using custom_categories.
        """
        try:
            result = self._memory.add(
                content,
                user_id=user_id,
                metadata=metadata or {},
                # NOTE: custom_categories is only supported in mem0 Platform API (MemoryClient),
                # not in the open-source version (Memory). For categorization, use:
                # 1. The custom_prompt to instruct LLM to include category in memory text
                # 2. Metadata field to store category manually
            )
            self.logger.info(
                "Added memory for user %s: %s",
                user_id,
                result,
            )
            return result
        except Exception as e:
            self.logger.error("Failed to add memory: %s", e)
            return {"error": str(e)}

    def search(
        self,
        query: str,
        user_id: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search memories by semantic similarity with optional category filter.

        Note: Categories are embedded in memory text as "[CATEGORY] content".
        We do semantic search first, then filter by category locally.
        """
        try:
            # Semantic search without category filter (category is in the text)
            # Request more results if filtering by category
            search_limit = limit * 3 if category else limit

            response = self._memory.search(
                query,
                user_id=user_id,
                limit=search_limit,
            )

            # mem0.search() returns {"results": [...], "relations": [...]}
            results = response.get("results", []) if isinstance(response, dict) else []

            # Filter by category if specified (category is embedded in text)
            if category and results:
                filtered = []
                for mem in results:
                    text = mem.get("memory", "")
                    mem_category, _ = parse_embedded_category(text)
                    if mem_category == category:
                        filtered.append(mem)
                results = filtered[:limit]  # Respect original limit

            self.logger.info(
                "Search for user %s, query='%s', category=%s: %d results",
                user_id,
                query[:50],
                category,
                len(results),
            )

            return results

        except Exception as e:
            self.logger.error("Failed to search memories: %s", e)
            return []

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get all memories for a user.
        """
        try:
            response = self._memory.get_all(user_id=user_id)
            # mem0.get_all() returns {"results": [...]}
            results = response.get("results", []) if isinstance(response, dict) else []
            self.logger.info(
                "Retrieved %d memories for user %s",
                len(results),
                user_id,
            )
            return results
        except Exception as e:
            self.logger.error("Failed to get all memories: %s", e)
            return []

    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.
        """
        try:
            self._memory.delete(memory_id)
            self.logger.info("Deleted memory %s", memory_id)
            return True
        except Exception as e:
            self.logger.error("Failed to delete memory %s: %s", memory_id, e)
            return False

    def delete_all(self, user_id: str) -> int:
        """
        Delete all memories for a user.
        """
        try:
            memories = self.get_all(user_id)
            deleted_count = 0

            for memory in memories:
                memory_id = memory.get("id")
                if memory_id and self.delete(memory_id):
                    deleted_count += 1

            self.logger.info(
                "Deleted %d memories for user %s",
                deleted_count,
                user_id,
            )
            return deleted_count

        except Exception as e:
            self.logger.error("Failed to delete all memories for %s: %s", user_id, e)
            return 0

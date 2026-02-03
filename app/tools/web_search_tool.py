"""
Web Search Tool for ADK Agent.

This tool uses a separate Gemini Flash call with native Google Search enabled
to perform web searches. This approach overcomes the limitation that Google Search
cannot be combined with other function calling tools in a single request.

The main agent can use this as a regular tool, while internally it executes
a dedicated LLM call with search grounding.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from google.genai import Client


# Module-level client instance (initialized lazily)
_client: Client | None = None
_logger = logging.getLogger(__name__)


def _get_client() -> Client:
    """
    Get or create the Google GenAI client.

    Uses Vertex AI backend with project/location from environment variables.

    Returns:
        Configured GenAI client.
    """
    global _client
    if _client is None:
        project_id = os.getenv("VERTEX_PROJECT_ID", "").strip() or None
        location = os.getenv("VERTEX_LOCATION", "global")
        _client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        _logger.info(
            "Initialized GenAI client for web search (project=%s, location=%s)",
            project_id,
            location,
        )
    return _client


async def web_search(query: str) -> dict:
    """
    Search the web for real-time information using Google Search grounding.

    This function performs a web search by making a dedicated Gemini Flash call
    with Google Search retrieval enabled. The search results are summarized
    and returned as structured data.

    Use this tool when you need current information that may not be in your
    training data, such as:
    - Current events and news
    - Recent product releases or updates
    - Real-time data (prices, weather, scores)
    - Verification of current facts

    Args:
        query: The search query to look up. Be specific and include relevant
               context for better results.

    Returns:
        A dictionary containing:
        - status: "success" or "error"
        - query: The original query
        - result: The search result summary (or error message)
        - sources: List of source URLs if available
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "query": query,
            "result": "Empty query provided",
            "sources": [],
        }

    try:
        client = _get_client()

        # Use Gemini Flash for fast, cost-effective search
        # The model is configured to use Google Search grounding
        search_model = os.getenv("SEARCH_MODEL_NAME", "gemini-3-flash-preview")

        # Configure the request with Google Search grounding enabled
        # Using dynamic retrieval for optimal results
        config = types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for factual search
            max_output_tokens=1024,
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch(),
                )
            ],
        )

        # Build the search prompt - instruct the model to search and summarize
        search_prompt = f"""Search the web for the following query and provide a comprehensive summary of the findings.

Query: {query}

Instructions:
1. Search for the most relevant and recent information
2. Summarize the key findings clearly
3. Include specific facts, numbers, or dates when available
4. Note if information might be outdated or uncertain"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=search_prompt)],
            )
        ]

        _logger.info("Executing web search for query: %s", query[:100])

        # Execute the search call
        response = await client.aio.models.generate_content(
            model=search_model,
            contents=contents,
            config=config,
        )

        if not response or not response.text:
            _logger.warning("Empty response from web search")
            return {
                "status": "error",
                "query": query,
                "result": "No results found for this query",
                "sources": [],
            }

        # Extract grounding metadata if available
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if hasattr(grounding, "grounding_chunks") and grounding.grounding_chunks:
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        sources.append(
                            {
                                "title": getattr(chunk.web, "title", ""),
                                "uri": getattr(chunk.web, "uri", ""),
                            }
                        )

        _logger.info(
            "Web search completed successfully. Found %d sources.", len(sources)
        )

        return {
            "status": "success",
            "query": query,
            "result": response.text,
            "sources": sources,
        }

    except Exception as e:
        _logger.error("Web search failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "query": query,
            "result": f"Search failed: {str(e)}",
            "sources": [],
        }

"""
Lightweight HTTP client for the Knowledge Core sidecar.

Provides a simple interface to interact with the knowledge base:
- health(): Check if the service is available
- add_fact(): Add a fact to the knowledge base
- query(): Query facts by pattern
- invalidate(): Invalidate (soft-delete) a fact
- register_schema(): Register a type schema for an attribute
"""

import logging
import os
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.services.KnowledgeService.knowledge_service_interface import (
    KnowledgeServiceInterface,
)


@runtime_checkable
class HttpClientProtocol(Protocol):
    """Protocol for HTTP clients (httpx.Client or mocks)."""

    def get(self, url: str, *, timeout: float | None = None) -> Any: ...
    def post(
        self, url: str, *, json: Any = None, timeout: float | None = None
    ) -> Any: ...


class KnowledgeService(KnowledgeServiceInterface):
    """
    HTTP client for the Knowledge Core sidecar.

    Configuration via constructor or environment:
        KNOWLEDGE_BASE_URL: Base URL for the service (default: http://127.0.0.1:8088)
    """

    def __init__(
        self,
        base_url: str | None = None,
        logger: logging.Logger | None = None,
        timeout: float = 2.0,
        http_client: HttpClientProtocol | None = None,
        # Health check tuning
        health_max_attempts: int = 3,
        health_base_backoff: float = 0.1,
        health_overall_timeout: float | None = None,
    ):
        env_url = os.getenv("KNOWLEDGE_BASE_URL")
        self.base_url = (base_url or env_url or "http://127.0.0.1:8088").rstrip("/")
        self.timeout = timeout
        self.logger = logger or logging.getLogger("knowledge_client")

        # Health tuning (kept for API compatibility)
        self.health_max_attempts = max(1, health_max_attempts)
        self.health_base_backoff = max(0.0, health_base_backoff)
        self.health_overall_timeout = health_overall_timeout

        # Use injected client or create a default one
        self._owns_client = http_client is None
        self._client: HttpClientProtocol = http_client or httpx.Client(timeout=timeout)

        self.logger.info(
            "KnowledgeClient initialized: base_url=%s timeout=%s",
            self.base_url,
            self.timeout,
        )

    def _make_retry_decorator(self, max_attempts: int = 3):
        """Create a retry decorator with exponential backoff and jitter."""
        return retry(
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=0.1, max=2.0, jitter=0.1),
            reraise=True,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """
        POST request with error handling. Returns parsed JSON or None on failure.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self.logger.warning(
                "KnowledgeClient POST %s failed with status %d: %s",
                path,
                e.response.status_code,
                e,
            )
            return None
        except httpx.RequestError as e:
            self.logger.warning("KnowledgeClient POST %s network error: %s", path, e)
            return None
        except Exception as e:
            self.logger.warning("KnowledgeClient POST %s unexpected error: %s", path, e)
            return None

    def _get(self, path: str) -> dict[str, Any] | None:
        """
        GET request with error handling. Returns parsed JSON or None on failure.
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.debug("KnowledgeClient GET %s failed: %s", path, e)
            return None

    def health(self) -> bool:
        """
        Check if the knowledge service is healthy.

        Retries with exponential backoff on transient failures.
        Returns True if healthy, False otherwise.
        """

        @self._make_retry_decorator(max_attempts=self.health_max_attempts)
        def _health_with_retry() -> bool:
            data = self._get("/health")
            if data is None:
                raise httpx.RequestError("Health check failed")

            # Accept either {"ok": true} or {"status": "ok"}
            if data.get("ok") is True or data.get("status") == "ok":
                return True
            raise httpx.RequestError("Unexpected health response")

        try:
            return _health_with_retry()
        except Exception:
            return False

    def add_fact(self, entity: str, attribute: str, value: Any) -> bool:
        """
        Add a fact to the knowledge base.

        Args:
            entity: The entity identifier (e.g., "user:123")
            attribute: The attribute name (e.g., "name", "age")
            value: The value (can be string, number, boolean, or JSON object)

        Returns:
            True if the fact was added successfully, False otherwise.
        """
        payload = {"entity": entity, "attribute": attribute, "value": value}
        data = self._post("/add-fact", payload)

        if data is None:
            return False

        ok = data.get("ok", False)
        if ok:
            self.logger.info(
                "KnowledgeClient: added fact id=%s attr=%s",
                data.get("id"),
                attribute,
            )
        return bool(ok)

    def query(self, pattern: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Query facts matching the given pattern.

        Args:
            pattern: Filter criteria (e.g., {"entity": "user:123", "attribute": "name"})

        Returns:
            List of matching facts, or empty list on error.
        """
        body = pattern if isinstance(pattern, dict) else {}
        data = self._post("/query", body)

        if data is None:
            return []

        facts = data.get("facts", [])
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, dict)]
        return []

    def invalidate(self, fact_id: str) -> bool:
        """
        Invalidate (soft-delete) a fact by ID.

        Args:
            fact_id: The fact identifier (e.g., "f-uuid")

        Returns:
            True if the fact was invalidated, False otherwise.
        """
        data = self._post("/invalidate-fact", {"id": fact_id})

        if data is None:
            return False

        return bool(data.get("ok", False))

    def register_schema(self, attribute: str, type_name: str) -> bool:
        """
        Register a type schema for an attribute.

        Args:
            attribute: The attribute name
            type_name: The type ("integer", "string", or "boolean")

        Returns:
            True if the schema was registered, False otherwise.
        """
        payload = {"attribute": attribute, "type": type_name}
        data = self._post("/register-schema", payload)

        if data is None:
            return False

        ok = data.get("ok", False)
        if ok:
            self.logger.info(
                "KnowledgeClient: registered schema attr=%s type=%s",
                attribute,
                type_name,
            )
        return bool(ok)

    def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client and self._client and hasattr(self._client, "close"):
            self._client.close()  # type: ignore[union-attr]

    def __enter__(self) -> "KnowledgeService":
        return self

    def __exit__(self, *args) -> None:
        self.close()

"""Shared retry configuration for Google GenAI API calls."""

import logging

from google.genai import errors as genai_errors
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

_logger = logging.getLogger(__name__)


def is_retryable_genai_error(exc: BaseException) -> bool:
    """Check if exception is a retryable Google GenAI API error (429, 502, or 503)."""
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    if isinstance(exc, genai_errors.ServerError):
        return getattr(exc, "code", None) in (502, 503)
    return False


genai_retry = retry(
    retry=retry_if_exception(is_retryable_genai_error),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=2),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)

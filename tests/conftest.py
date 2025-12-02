"""
Pytest configuration and fixtures for the test suite.

This file is automatically loaded by pytest before running tests.
It disables Langfuse tracing to prevent sending traces during test runs.
"""

import os
import logging

# Disable Langfuse tracing before any test modules import Langfuse
# This must be set BEFORE langfuse is imported
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

# Silence Langfuse logger immediately at module load time
# This needs to happen BEFORE pytest's logging is configured
_langfuse_logger = logging.getLogger("langfuse")
_langfuse_logger.setLevel(logging.CRITICAL)
_langfuse_logger.propagate = False

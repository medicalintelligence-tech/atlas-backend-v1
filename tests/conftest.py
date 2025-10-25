"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Logging configuration for tests
- Shared fixtures available to all tests
"""

import pytest
from src.config.logging import setup_logging


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """
    Configure logging for all tests.

    This fixture runs once per test session and enables logging output
    during test execution. Set to DEBUG level to see detailed logs from
    services like Azure OCR.
    """
    # Use DEBUG level to see all logs during tests
    # Change to INFO or WARNING if too verbose
    setup_logging(level="DEBUG", format_style="detailed")
    yield


@pytest.fixture(scope="session")
def configure_logging_info():
    """
    Alternative logging configuration at INFO level.

    Use this fixture explicitly in tests where you want less verbose output:
        def test_something(configure_logging_info):
            ...
    """
    setup_logging(level="INFO", format_style="detailed")
    yield

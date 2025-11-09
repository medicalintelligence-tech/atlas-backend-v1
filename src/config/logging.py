"""
Logging configuration for the application.

Provides centralized logging setup with different configurations for:
- Development (verbose, colorful console output)
- Production (structured JSON logs)
- Testing (configurable verbosity)
"""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    format_style: Literal["simple", "detailed"] = "detailed",
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Output format style
            - "simple": Just the message
            - "detailed": Timestamp, level, module, and message
    """
    # Define format strings
    formats = {
        "simple": "%(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        format=formats[format_style],
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )

    # Set specific loggers to appropriate levels
    # Reduce noise from third-party libraries
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, format={format_style}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Usually __name__ from the calling module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

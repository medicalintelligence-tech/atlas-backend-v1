from src.config.logging import setup_logging, get_logger

# Configure logging when module is imported
setup_logging(level="INFO", format_style="detailed")

logger = get_logger(__name__)


def main():
    logger.info("Starting backend application")
    print("Hello from backend!")
    logger.info("Backend application completed")


if __name__ == "__main__":
    main()

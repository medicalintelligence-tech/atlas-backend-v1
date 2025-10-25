from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Can be loaded from different .env files for different environments:
    - .env (local development, gitignored)
    - .env.integration (integration tests)
    - .env.production (production, never committed)
    """

    # Azure Document Intelligence
    azure_document_intelligence_endpoint: str | None = None
    azure_document_intelligence_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Create a global instance for importing
settings = Settings()
